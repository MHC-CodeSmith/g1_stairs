"""Load a DWAQ checkpoint into a policy whose actor obs gained `extra` dims appended at the end (e.g. a dance clock).

New input columns / decoder outputs are zero-initialized, so the expanded network starts out computing exactly what
the source policy computed and learns to use the new inputs. Layouts (rsl_rl ActorCritic_DWAQ + G1DanceEnv):
  actor.0   input = [latent code (19), obs]                       -> new columns at the end
  critic.0  input = [obs, privileged..., height scan]             -> new columns right after obs
  encoder.0 input = obs history, frame-major (hist x obs)         -> new columns at the end of every frame
  decoder.4 output = reconstructed obs                            -> new rows at the end
"""
import torch


def load_mapped(policy, src_state: dict, obs_map: list[int | None], hist_len: int) -> list[str]:
    """Load a DWAQ checkpoint into a policy whose actor observation is a re-arrangement of the source's.

    obs_map[i] = source obs index feeding new obs index i, or None for a new input (zero weights / zero decoder
    row). Applies to the actor input (after the 19-dim code), the critic's leading actor block (the privileged
    tail is kept in order), every encoder history frame and the decoder output. Source dims absent from the map are
    dropped (e.g. a dance clock the new task doesn't have).
    """
    dst = policy.state_dict()
    new_obs = dst["decoder.4.weight"].shape[0]
    old_obs = src_state["decoder.4.weight"].shape[0]
    assert len(obs_map) == new_obs
    code = src_state["actor.0.weight"].shape[1] - old_obs
    out, changed = {}, []
    for k, v in src_state.items():
        want = dst[k].shape
        if k == "actor.0.weight":
            w = torch.zeros(want, dtype=v.dtype)
            w[:, :code] = v[:, :code]
            for i, j in enumerate(obs_map):
                if j is not None:
                    w[:, code + i] = v[:, code + j]
        elif k == "critic.0.weight":
            w = torch.zeros(want, dtype=v.dtype)
            for i, j in enumerate(obs_map):
                if j is not None:
                    w[:, i] = v[:, j]
            assert v.shape[1] - old_obs == want[1] - new_obs, "privileged tail length differs"
            w[:, new_obs:] = v[:, old_obs:]
        elif k == "encoder.0.weight":
            w = torch.zeros(want, dtype=v.dtype)
            for f in range(hist_len):
                for i, j in enumerate(obs_map):
                    if j is not None:
                        w[:, f * new_obs + i] = v[:, f * old_obs + j]
        elif k in ("decoder.4.weight", "decoder.4.bias"):
            w = torch.zeros(want, dtype=v.dtype)
            for i, j in enumerate(obs_map):
                if j is not None:
                    w[i] = v[j]
        else:
            w = v
        if tuple(w.shape) != tuple(want):
            raise ValueError(f"don't know how to map {k}: {tuple(v.shape)} -> {tuple(want)}")
        if w is not v:
            changed.append(f"{k} {tuple(v.shape)}->{tuple(want)}")
        out[k] = w
    policy.load_state_dict(out)
    return changed


def load_expanded(policy, src_state: dict, old_obs: int, hist_len: int) -> list[str]:
    dst = policy.state_dict()
    new_obs = dst["decoder.4.weight"].shape[0]
    extra = new_obs - old_obs
    out, changed = {}, []
    for k, v in src_state.items():
        want = dst[k].shape
        if v.shape == want:
            out[k] = v
            continue
        w = torch.zeros(want, dtype=v.dtype)
        if k == "actor.0.weight":
            w[:, : v.shape[1]] = v
        elif k == "critic.0.weight":
            w[:, :old_obs] = v[:, :old_obs]
            w[:, old_obs + extra:] = v[:, old_obs:]
        elif k == "encoder.0.weight":
            for f in range(hist_len):
                w[:, f * new_obs: f * new_obs + old_obs] = v[:, f * old_obs:(f + 1) * old_obs]
        elif k in ("decoder.4.weight", "decoder.4.bias"):
            w[: v.shape[0]] = v
        else:
            raise ValueError(f"don't know how to expand {k}: {tuple(v.shape)} -> {tuple(want)}")
        out[k] = w
        changed.append(f"{k} {tuple(v.shape)}->{tuple(want)}")
    policy.load_state_dict(out)
    return changed
