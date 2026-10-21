"""Compile the shared SAM2 refinement decoder before deployment, using reference inputs only."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import struct
import zipfile
import time

import numpy as np
from PIL import Image
import torch

from transformers import Sam2Model, Sam2Processor

from gsedit_scd.online.cues import SAM_MODEL
from gsedit_scd.online.refinement import preprocess_gpu

class Decoder(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, image, position, sparse, dense, high0, high1):
        return self.decoder(image_embeddings=image, image_positional_embeddings=position,
                            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                            high_resolution_features=[high0, high1], multimask_output=True)


def prepare_library(package, destination):
    """Extract runtime files and clear the build toolchain's executable-stack flag."""
    destination.mkdir()
    with zipfile.ZipFile(package) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if not (name.endswith('.wrapper.so') or name.endswith('.cubin')):
                continue
            data = bytearray(archive.read(member))
            if name.endswith('.so'):
                # The kernels need no executable CPU stack. This is the same ELF
                # metadata correction used for the validated Ubuntu AOT artifacts.
                if data[:6] != b'\x7fELF\x02\x01':
                    raise ValueError('Expected a little-endian ELF64 shared library')
                offset = struct.unpack_from('<Q', data, 32)[0]
                entry_size, count = struct.unpack_from('<HH', data, 54)
                for index in range(count):
                    at = offset + index * entry_size
                    kind, flags = struct.unpack_from('<II', data, at)
                    if kind == 0x6474e551:  # PT_GNU_STACK
                        struct.pack_into('<I', data, at + 4, flags & ~1)
            (destination / name).write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--events', type=Path,
                        help='Reference-only training_events.tsv used to obtain example inputs.')
    inputs.add_argument('--reference-image', type=Path,
                        help='Reference RGB for compilation with pretrained parameters; no edits or labels needed.')
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory for compiled packages and prepared runtime files.')
    options = parser.parse_args()
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    print('ENVIRONMENT', Path.cwd(), sys.executable, torch.__version__, os.environ.get('CXX'), flush=True)
    model = Sam2Model.from_pretrained(SAM_MODEL, local_files_only=True, dtype=torch.float16).eval().cuda()
    model.requires_grad_(False)
    processor = Sam2Processor.from_pretrained(SAM_MODEL, local_files_only=True)
    if options.events:
        with options.events.open() as stream:
            row = next(csv.DictReader(stream, delimiter='\t'))
        reference_image, reference_edit_mask = row['reference_image'], row['positive_mask']
    else:
        reference_image, reference_edit_mask = str(options.reference_image), None
    with Image.open(reference_image) as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    if reference_edit_mask is not None:
        with Image.open(reference_edit_mask) as image:
            mask = np.asarray(image.convert('L')) > 0
        ys, xs = np.nonzero(mask)
        box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        point = [float(xs[len(xs)//2]), float(ys[len(ys)//2])]
    else:
        # These prompts supply compiler example inputs, not training targets.
        height, width = rgb.shape[:2]
        box = [0.0, 0.0, float(width - 1), float(height - 1)]
        point = [(width - 1) / 2, (height - 1) / 2]
    captured = []
    def captured_inputs(module, args, kwargs):
        captured.clear()
        captured.extend([kwargs[k].contiguous() for k in ('image_embeddings', 'image_positional_embeddings',
                        'sparse_prompt_embeddings', 'dense_prompt_embeddings')])
        captured.extend(x.contiguous() for x in kwargs['high_resolution_features'])
    results = {}
    wrapper = Decoder(model.mask_decoder).eval()
    with torch.inference_mode():
        target = torch.from_numpy(rgb).cuda()
        pixels, sizes = preprocess_gpu(processor, target, target.permute(2, 0, 1).float()/255)
        embeddings = model.get_image_embeddings(pixels.to(dtype=model.dtype))
        for kind in ('box', 'point'):
            for case, count in [('single', 1), ('multiple', 3)]:
                name = kind+'_'+case
                if kind == 'box':
                    prompts = processor(original_sizes=sizes, input_boxes=[[box]*count, [box]*count], return_tensors='pt')
                    inputs = dict(image_embeddings=embeddings, input_boxes=prompts['input_boxes'].cuda())
                else:
                    prompts = processor(original_sizes=sizes[:1], input_points=[[[point] for _ in range(count)]],
                                                input_labels=[[[1] for _ in range(count)]], return_tensors='pt')
                    inputs = dict(image_embeddings=[x[:1] for x in embeddings], input_points=prompts['input_points'].cuda(),
                                  input_labels=prompts['input_labels'].cuda())
                handle = model.mask_decoder.register_forward_pre_hook(captured_inputs, with_kwargs=True)
                model(multimask_output=True, **inputs)
                handle.remove()
                args = tuple(captured)
                shape_spec = ({}, {}, {1: torch.export.Dim('prompts', min=2, max=64)} if count > 1 else {}, {}, {}, {})
                print('EXPORT', name, [list(x.shape) for x in args], flush=True)
                started = time.perf_counter()
                exported = torch.export.export(wrapper, args, dynamic_shapes=shape_spec)
                exported_seconds = time.perf_counter()-started
                print('COMPILE', name, flush=True)
                path = torch._inductor.aoti_compile_and_package(exported, package_path=str(output/(name+'.pt2')))
                torch.cuda.synchronize()
                prepare_library(Path(path), output/name)
                results[name] = dict(path=str(path), export_seconds=exported_seconds,
                                     total_seconds=time.perf_counter()-started, bytes=Path(path).stat().st_size,
                                     input_shapes=[list(x.shape) for x in args])
                (output/'build_results.json').write_text(json.dumps(dict(reference_image=reference_image,
                    reference_edit_mask=reference_edit_mask,numerics='TF32 false, FP16 SAM; contiguous decoder inputs',
                    torch_version=torch.__version__,models=results),indent=2)+'\n')
                print('BUILT', name, results[name], flush=True)
    print('COMPLETE', flush=True)


if __name__ == "__main__":
    main()
