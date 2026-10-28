(*---------------------------------------------------------------------------
  Copyright (c) 2026 The geom authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A module's structure constants are host values. [beside x c] is [c] placed
   whole on each device of [x]'s placement, so that an operation of [x] reads
   it; on the host it is [c] itself, without a copy. *)
let beside x c = Nx.place (Nx.Placement.uncut (Nx.placement x)) c

(* [last x i] is element [i] of [x]'s last axis, a view. *)
let last x i =
  let r = Array.length (Nx.shape x) in
  if r = 0 then invalid_arg "geom: a value needs a last axis";
  Nx.slice (List.init (pred r) (fun _ -> Nx.A) @ [ Nx.I i ]) x
