(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Ilqr_vae authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Base
open Nx

type ('a, 'd) params =
  { d : 'a
  ; t : 'a
  }

let walk_params cur x =
  let open Nx.Ptree.Walk in
  let d = field cur "d" leaf x.d in
  let t = field cur "t" leaf x.t in
  { d; t }

let structure () =
  Nx.Ptree.instantiate
    (module struct
      type ('a, 'd) t = ('a, 'd) params

      let walk = walk_params
    end)

let make ?(sigma2 = 1.0) ~n dtype =
  if n <= 0 then invalid_arg "Covariance.make: n must be positive";
  if not (Float.( > ) sigma2 0.0)
  then invalid_arg "Covariance.make: sigma2 must be positive";
  { d = full dtype [| n |] (Float.sqrt sigma2); t = zeros dtype [| n; n |] }

let factor { d; t } = Infix.(triu ~k:1 t + diag d)
let log_det { d; _ } = Infix.(sum (log d) *$ 2.0)
