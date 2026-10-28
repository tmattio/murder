(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx
open Types

type ('w, 'p, 'd) t =
  { n : int
  ; m : int
  ; structure : ('p, 'd) Nx.Ptree.t
  ; params : 'p
  ; dyn : 'p -> k:('w, 'd) value -> x:('w, 'd) value -> u:('w, 'd) value -> ('w, 'd) value
  }

let integrate (d : ('w, 'p, 'd) t) (p : 'p) ~(x0 : ('w, 'd) value) ~(u : ('w, 'd) value) =
  let horizon = (shape u).(0) in
  let at = Nx.Placement.uncut (Nx.placement x0) in
  let ks = arange_at at (Nx.dtype x0) 0 horizon 1 in
  let _, xs =
    Rune.scan
      Nx.Ptree.tensor
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      ~f:(fun x (u_k, k) ->
        let x' = d.dyn p ~k ~x ~u:u_k in
        x', x')
      ~init:x0
      (u, ks)
  in
  (* [x0] may be one state or a batch of states; broadcast it to the carry's
     batch shape before stacking the initial state in front of the path. *)
  let batch_shape = Array.sub (shape xs) ~pos:1 ~len:(Array.length (shape xs) - 1) in
  let x0 = reshape (Array.append [| 1 |] batch_shape) (broadcast_to batch_shape x0) in
  concatenate ~axis:0 [ x0; xs ]

(* [randn ?key dtype shape] is a standard normal draw, from [key] when given
   and from the implicit random scope otherwise. *)
let randn ?key dtype shape = Nx.Rng.normal ?key dtype shape

(* [scaled_randn ?key ~sigma dtype shape] is a normal draw of standard
   deviation [sigma]. *)
let scaled_randn ?key ~sigma dtype shape = Infix.(randn ?key dtype shape *$ sigma)

(* [first_step k] is a boolean scalar that is [true] at step 0. A dynamics
   written for these constructors selects on it with [Nx.where], since [k] is
   a tensor under the encoder's scans. *)
let first_step (k : ('w, 'd) value) = Nx.equal k (scalar_like k 0.0)

(* [input_term ~dt_over_tau ~k ~u] is the input contribution of one step: the
   projected input itself at the first step, [dt_over_tau] times it
   afterwards. Applying the whole scaled input at step 0 is the convention of
   the reference implementation. *)
let input_term ~dt_over_tau ~k ~u =
  let scaled =
    match dt_over_tau with
    | `Scalar dt -> Infix.(u *$ dt)
    | `Vector dt -> Infix.(u * dt)
  in
  where (first_step k) u scaled

(* {1 Linear} *)

module Linear = struct
  type ('a, 'd) params =
    { dt_over_tau : float
    ; w : 'a
    ; b : 'a option
    }

  let walk_params cur x =
    let open Nx.Ptree.Walk in
    let dt_over_tau = x.dt_over_tau in
    let w = field cur "w" leaf x.w in
    let b = field cur "b" (option leaf) x.b in
    { dt_over_tau; w; b }

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 'd) t = ('a, 'd) params

        let walk = walk_params
      end)

  let matrix { dt_over_tau; w; _ } =
    let identity =
      eye_at (Nx.Placement.uncut (Nx.placement w)) (Nx.dtype w) (Array.get (shape w) 0)
    in
    Infix.(identity + ((w - identity) *$ dt_over_tau))

  let dyn p ~k ~x ~u =
    let u =
      match p.b with
      | None -> u
      | Some b -> Infix.(u *@ b)
    in
    let input = input_term ~dt_over_tau:(`Scalar p.dt_over_tau) ~k ~u in
    Infix.((x *@ matrix p) + input)

  let make ?key ?(alpha = 1.0) ~n ~m dtype =
    if n <= 0 then invalid_arg "Dynamics.Linear.make: n must be positive";
    if m <= 0 then invalid_arg "Dynamics.Linear.make: m must be positive";
    let w = Infix.(eye dtype n *$ alpha) in
    let b = if Int.equal n m then None else Some (randn ?key dtype [| m; n |]) in
    { n; m; structure = structure (); params = { dt_over_tau = 0.1; w; b }; dyn }
end

(* {1 Linear–nonlinear} *)

module Linear_nonlinear = struct
  type ('a, 'd) params =
    { dt_over_tau : 'a
    ; a0 : 'a
    ; a1 : 'a
    ; a2 : 'a
    ; bias1 : 'a
    ; bias2 : 'a
    ; b : 'a option
    }

  let walk_params cur x =
    let open Nx.Ptree.Walk in
    let dt_over_tau = field cur "dt_over_tau" leaf x.dt_over_tau in
    let a0 = field cur "a0" leaf x.a0 in
    let a1 = field cur "a1" leaf x.a1 in
    let a2 = field cur "a2" leaf x.a2 in
    let bias1 = field cur "bias1" leaf x.bias1 in
    let bias2 = field cur "bias2" leaf x.bias2 in
    let b = field cur "b" (option leaf) x.b in
    { dt_over_tau; a0; a1; a2; bias1; bias2; b }

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 'd) t = ('a, 'd) params

        let walk = walk_params
      end)

  let dyn ~phi p ~k ~x ~u =
    let hidden = phi Infix.((x *@ p.a1) + p.bias1) in
    let passive =
      Infix.(x + (((x *@ p.a0) - x + ((hidden *@ p.a2) + p.bias2)) * p.dt_over_tau))
    in
    let u =
      match p.b with
      | None -> u
      | Some b -> Infix.(u *@ b)
    in
    Infix.(passive + input_term ~dt_over_tau:(`Vector p.dt_over_tau) ~k ~u)

  let make ?key ?(radius = 0.1) ~dt ~tau ~n ~nh ~m ~phi dtype =
    if n <= 0 then invalid_arg "Dynamics.Linear_nonlinear.make: n must be positive";
    if nh <= 0 then invalid_arg "Dynamics.Linear_nonlinear.make: nh must be positive";
    if m <= 0 then invalid_arg "Dynamics.Linear_nonlinear.make: m must be positive";
    if not (Float.( > ) dt 0.0 && Float.( > ) tau 0.0)
    then invalid_arg "Dynamics.Linear_nonlinear.make: dt and tau must be positive";
    let params =
      { dt_over_tau = full dtype [| n |] (dt /. tau)
      ; a0 = Infix.(eye dtype n *$ 0.5)
      ; a1 =
          scaled_randn ?key ~sigma:(1.0 /. Float.sqrt (Float.of_int n)) dtype [| n; nh |]
      ; a2 =
          scaled_randn
            ?key
            ~sigma:(radius /. Float.sqrt (Float.of_int nh))
            dtype
            [| nh; n |]
      ; bias1 = Infix.((Nx.Rng.uniform dtype [| nh |] *$ 2.0) -$ 1.0)
      ; bias2 = zeros dtype [| n |]
      ; b =
          Some
            (scaled_randn
               ?key
               ~sigma:(1.0 /. Float.sqrt (Float.of_int m))
               dtype
               [| m; n |])
      }
    in
    { n
    ; m
    ; structure = structure ()
    ; params
    ; dyn = (fun p ~k ~x ~u -> dyn ~phi p ~k ~x ~u)
    }
end

(* {1 Gated nonlinear} *)

module Gnode = struct
  type ('a, 'd) params =
    { dt_over_tau : float
    ; g_a1 : 'a
    ; g_bias1 : 'a
    ; h_a1 : 'a
    ; h_a2 : 'a
    ; h_bias1 : 'a
    ; h_bias2 : 'a
    ; b : 'a option
    }

  let walk_params cur x =
    let open Nx.Ptree.Walk in
    let dt_over_tau = x.dt_over_tau in
    let g_a1 = field cur "g_a1" leaf x.g_a1 in
    let g_bias1 = field cur "g_bias1" leaf x.g_bias1 in
    let h_a1 = field cur "h_a1" leaf x.h_a1 in
    let h_a2 = field cur "h_a2" leaf x.h_a2 in
    let h_bias1 = field cur "h_bias1" leaf x.h_bias1 in
    let h_bias2 = field cur "h_bias2" leaf x.h_bias2 in
    let b = field cur "b" (option leaf) x.b in
    { dt_over_tau; g_a1; g_bias1; h_a1; h_a2; h_bias1; h_bias2; b }

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 'd) t = ('a, 'd) params

        let walk = walk_params
      end)

  let dyn ~phi p ~k ~x ~u =
    let g = phi Infix.((x *@ p.g_a1) + p.g_bias1) in
    let h = Infix.((phi ((x *@ p.h_a1) + p.h_bias1) *@ p.h_a2) + p.h_bias2) in
    let passive = Infix.(x + (g * (h - x) *$ p.dt_over_tau)) in
    let u =
      match p.b with
      | None -> u
      | Some b -> Infix.(u *@ b)
    in
    Infix.(passive + input_term ~dt_over_tau:(`Scalar p.dt_over_tau) ~k ~u)

  let make ?key ?(radius = 0.1) ?m ~dt ~tau ~n ~nh ~phi dtype =
    if n <= 0 then invalid_arg "Dynamics.Gnode.make: n must be positive";
    if nh <= 0 then invalid_arg "Dynamics.Gnode.make: nh must be positive";
    if not (Float.( > ) dt 0.0 && Float.( > ) tau 0.0)
    then invalid_arg "Dynamics.Gnode.make: dt and tau must be positive";
    let m_input = m in
    let m = Option.value m ~default:n in
    let params =
      { dt_over_tau = dt /. tau
      ; g_a1 = zeros dtype [| n; n |]
      ; g_bias1 = zeros dtype [| n |]
      ; h_a1 =
          scaled_randn ?key ~sigma:(1.0 /. Float.sqrt (Float.of_int n)) dtype [| n; nh |]
      ; h_a2 =
          scaled_randn
            ?key
            ~sigma:(radius /. Float.sqrt (Float.of_int nh))
            dtype
            [| nh; n |]
      ; h_bias1 = zeros dtype [| nh |]
      ; h_bias2 = zeros dtype [| n |]
      ; b = Option.map m_input ~f:(fun m -> randn ?key dtype [| m; n |])
      }
    in
    { n
    ; m
    ; structure = structure ()
    ; params
    ; dyn = (fun p ~k ~x ~u -> dyn ~phi p ~k ~x ~u)
    }
end

(* {1 Minimal gated unit} *)

module Mgu = struct
  type ('a, 'd) params =
    { uf : 'a
    ; uh : 'a
    ; w : 'a
    ; bf : 'a
    ; bh : 'a
    }

  let walk_params cur x =
    let open Nx.Ptree.Walk in
    let uf = field cur "uf" leaf x.uf in
    let uh = field cur "uh" leaf x.uh in
    let w = field cur "w" leaf x.w in
    let bf = field cur "bf" leaf x.bf in
    let bh = field cur "bh" leaf x.bh in
    { uf; uh; w; bf; bh }

  let structure () =
    Nx.Ptree.instantiate
      (module struct
        type ('a, 'd) t = ('a, 'd) params

        let walk = walk_params
      end)

  (* The paper's Appendix B.2 (S7)-(S9), with the [t - 1] subscript typo of
     (S9) corrected and the biases it omits made explicit. [b_f] is not
     optional: at the encoder's zero initial state a biasless gate is
     [phi 0 = 0] and the state never leaves the origin. *)
  let dyn ~phi p ~k:_ ~x ~u =
    let f = phi Infix.((x *@ p.uf) + p.bf) in
    let h = Infix.(phi ((f * x *@ p.uh) + p.bh) + (u *@ p.w)) in
    Infix.(((ones_like f - f) * x) + (f * h))

  let make ?key ?(bias_f = 0.5) ~n ~m ~phi dtype =
    if n <= 0 then invalid_arg "Dynamics.Mgu.make: n must be positive";
    if m <= 0 then invalid_arg "Dynamics.Mgu.make: m must be positive";
    let params =
      { uf = zeros dtype [| n; n |]
      ; uh = zeros dtype [| n; n |]
      ; w = scaled_randn ?key ~sigma:(1.0 /. Float.sqrt (Float.of_int m)) dtype [| m; n |]
      ; bf = full dtype [| n |] bias_f
      ; bh = zeros dtype [| n |]
      }
    in
    { n
    ; m
    ; structure = structure ()
    ; params
    ; dyn = (fun p ~k ~x ~u -> dyn ~phi p ~k ~x ~u)
    }
end
