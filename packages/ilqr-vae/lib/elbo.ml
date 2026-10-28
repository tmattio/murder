(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Types

type ('w, 'd) t =
  { elbo : ('w, 'd) value
  ; log_likelihood : ('w, 'd) value
  ; kl : ('w, 'd) value
  }

(* [mean_over_batch t n] is the mean of a [[n]] tensor of per-sample
   scalars. *)
let mean_over_batch t n = Infix.(sum t /$ Float.of_int n)

(* [log_prior model params ~x0 ~u] is the log density of the model's prior
   over the latents at a sample: one value per member of the batch. The input
   prior is evaluated step by step; the initial-state prior contributes
   exactly when the model has one. *)
let log_prior
      (type w y l p e r)
      (model : (w, y, l, p, e, r, 'd) Model.t)
      (params : (w, y, l, p, e, r, 'd) Model.params)
      ~(x0 : (w, 'd) value)
      ~(u : (w, 'd) value)
  =
  let t_steps = model.Model.n_steps in
  let at = Nx.Placement.uncut (Nx.placement x0) in
  let ks = arange_at at model.Model.dtype 0 t_steps 1 in
  let _, by_step =
    Rune.scan
      Nx.Ptree.unit
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      ~f:(fun () (u_k, k) ->
        (), model.Model.prior.Prior.logp_t params.Model.prior ~k ~u:u_k)
      ~init:()
      (u, ks)
  in
  let u_logp = sum ~axes:[ 0 ] by_step in
  match params.Model.initial_prior with
  | None -> u_logp
  | Some p -> Infix.(u_logp + Initial.logp p x0)

let compute
      (type w y l p e r)
      (model : (w, y, l, p, e, r, 'd) Model.t)
      (params : (w, y, l, p, e, r, 'd) Model.params)
      ?beta
      ~(posterior : (w, 'd) Recognition.posterior)
      data
  =
  let beta = Option.value beta ~default:(scalar_like posterior.Recognition.x0 1.0) in
  let t_steps = model.Model.n_steps in
  let n_samples = Array.get (shape posterior.Recognition.x0) 0 in
  if n_samples <= 0 then invalid_arg "Elbo.compute: the posterior batch is empty";
  let z =
    Dynamics.integrate
      model.Model.dynamics
      params.Model.dynamics
      ~x0:posterior.Recognition.x0
      ~u:posterior.Recognition.u
  in
  (* Observation [k] is emitted from the state reached by [k + 1] inputs. *)
  let z_next = slice [ R (1, t_steps + 1) ] z in
  let at = Nx.Placement.uncut (Nx.placement z) in
  let ks = arange_at at model.Model.dtype 0 t_steps 1 in
  let _, likelihood_by_step =
    Rune.scan
      Nx.Ptree.unit
      Nx.Ptree.(pair (pair tensor tensor) tensor)
      Nx.Ptree.tensor
      ~f:(fun () ((z_k, o_k), k) ->
        ( ()
        , model.Model.likelihood.Likelihood.logp_t
            params.Model.likelihood
            ~k
            ~x:z_k
            ~o:o_k ))
      ~init:()
      ((z_next, data), ks)
  in
  let log_likelihood_by_sample = sum ~axes:[ 0 ] likelihood_by_step in
  let log_prior_by_sample =
    log_prior model params ~x0:posterior.Recognition.x0 ~u:posterior.Recognition.u
  in
  (* The sampled KL, and the bound at [beta]. Both densities are evaluated at
     the same reparameterized draws; {!Recognition.infer} stopped the
     recognition's parameters inside [log_prob], so differentiating this is
     the path-derivative (sticking-the-landing) estimator of Roeder, Wu &
     Duvenaud (2017): the gradient reaches the recognition through the draws
     only. *)
  let kl_by_sample = Infix.(posterior.Recognition.log_prob - log_prior_by_sample) in
  let elbo_by_sample = Infix.(log_likelihood_by_sample - (kl_by_sample * beta)) in
  { elbo = mean_over_batch elbo_by_sample n_samples
  ; log_likelihood = mean_over_batch log_likelihood_by_sample n_samples
  ; kl = mean_over_batch kl_by_sample n_samples
  }

let loss model params ?beta ?iters ~key ~n_samples data =
  let posterior = Model.infer model params ?iters ~n_samples ~key data in
  Infix.(-(compute model params ?beta ~posterior data).elbo)

let loss_fixed model params ?beta ~iters ~key ~n_samples data =
  loss model params ?beta ~iters ~key ~n_samples data

let batch_loss_fixed model params ?beta ~key ~n_samples ~iters trials =
  let count = Array.length trials in
  if count = 0 then invalid_arg "Elbo.batch_loss_fixed: the batch is empty";
  let keys = Nx.Rng.split ~n:count key in
  let total =
    Array.mapi trials ~f:(fun i data ->
      loss_fixed model params ?beta ~key:keys.(i) ~n_samples ~iters data)
    |> Array.reduce_exn ~f:Infix.( + )
  in
  Infix.(total /$ Float.of_int count)
