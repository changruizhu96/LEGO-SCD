"""Query prediction followed by one update of the Gaussian support field."""
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from gsedit_scd.vendor.oscd.gaussian_renderer import render
from gsedit_scd.vendor.oscd.scene.cameras import Camera
from gsedit_scd.vendor.oscd.scene.gaussian_model import GaussianModel
from .evidence import (SupportField, decode_probability,
                       fusion_score, neighbourhood_log_bayes_factor)
from .frontend import current_evidence
from .geometry import (aggregate_primitive_measurement, build_gaussian_overlap_graph,
                       render_probability_map)


@dataclass(frozen=True)
class Ablation:
    suppression: bool = True
    mapping: bool = True
    geometry: bool = True
    history: bool = True


def load_scene(ply_path):
    gaussians = GaussianModel(3, 3)
    gaussians.load_ply(str(ply_path))
    return gaussians


def cached_view(payload, height, width, gaussians, pipe, background, *, with_measurement=True):
    name = payload["image_name"]
    view = Camera(colmap_id=name, R=payload["R"].numpy(), T=payload["T"].numpy(),
                  FoVx=float(payload["FoVx"]), FoVy=float(payload["FoVy"]),
                  image=torch.zeros((3, height, width), device=background.device),
                  gt_alpha_mask=None, image_name=name, uid=name)
    if with_measurement:
        view.probability_measurement = payload["candidate_map"].to(background.device).detach().clone()
    with torch.no_grad():
        view.reference_render = render(view, gaussians, pipe, background)["render"].detach()
    return view


class OnlineDetector:
    def __init__(self, gaussians, suppression, log_likelihood_ratio, alpha, beta,
                 spatial_correlation, prior_log_odds, ablation=Ablation(), *,
                 baseline_mean=None, baseline_variance=None):
        self.gaussians = gaussians
        self.pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        self.background = torch.zeros(3, device=gaussians.get_xyz.device)
        self.suppression = suppression
        self.log_likelihood_ratio = log_likelihood_ratio
        self.graph = self.mean = self.variance = self.field = None
        if ablation.history:
            self.graph, _ = build_gaussian_overlap_graph(gaussians, 8, appearance_aware=True)
            if baseline_mean is not None:
                self.mean = baseline_mean.to(self.background.device).float()
                self.variance = baseline_variance.to(self.background.device).float()
            else:
                # Retained positive-strength checkpoints store Beta parameters.
                concentration = alpha.float() + beta.float()
                self.mean = alpha.float() / concentration
                self.variance = alpha.float() * beta.float() / (concentration.square() * (concentration + 1.0)).clamp_min(1e-8)
            self.field = SupportField(len(self.mean), gaussians.get_xyz.device)
        self.c = spatial_correlation
        self.prior_log_odds = prior_log_odds
        self.ablation = ablation

    def predict_then_update(self, view, *, source_replay=False):
        """Inactive history has no field or per-Gaussian diagnostic tensors."""
        current = current_evidence(
            view, self.gaussians, self.pipe, self.background,
            self.suppression, self.log_likelihood_ratio,
            use_suppression=self.ablation.suppression,
            use_mapping=self.ablation.mapping, use_geometry=self.ablation.geometry,
        )
        if not self.ablation.history or self.field.frame_count == 0:
            history = torch.full_like(current.combined, 0.5)
        elif source_replay:
            # The retained source replay reads the first rendered channel.
            colors = self.field.read()[:, None].expand(-1, 3).contiguous()
            mass = render(view, self.gaussians, self.pipe, self.background,
                          override_color=colors)["render"][0]
            history = torch.where(current.coverage > 0.001,
                                  (mass / current.coverage.clamp_min(1e-6)).clamp(0, 1), 0.0)
        else:
            history = render_probability_map(view, self.gaussians, self.pipe, self.background,
                                             self.field.read(), current.coverage)
        covered = current.coverage.detach().squeeze().clamp(0.0, 1.0).double() > 0.001
        update_probability = current.combined
        score = fusion_score(current.combined.detach(), history.detach(), covered)
        probability = torch.where(covered, decode_probability(score, self.prior_log_odds), 0.0)

        # Predictions use only the previous field. Current evidence enters the
        # field after decoding; output refinement never enters this update.
        log_bf = weight = None
        if self.ablation.history:
            with torch.enable_grad():
                response, variance, support, _ = aggregate_primitive_measurement(
                    view, self.gaussians, self.pipe, self.background, update_probability
                )
            weight = (support >= 1).to(support.dtype)
            log_bf = neighbourhood_log_bayes_factor(response, variance, weight, self.graph,
                                                   self.mean, self.variance, self.c)
            self.field.update(log_bf, weight)
        return {"probability": probability.squeeze(), "score": score.squeeze(),
                "current": current.combined.squeeze(), "history": history.squeeze(),
                "coverage": current.coverage.squeeze(),
                "covered": covered.squeeze(),
                "log_bayes_factor": log_bf, "observation_weight": weight}
