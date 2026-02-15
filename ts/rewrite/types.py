from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import onnx_graphsurgeon as gs

HeightRange = Tuple[int, int]


@dataclass(frozen=True)
class ConvSlice:
    slice_start: int
    slice_end: int
    pad_top: int
    pad_bottom: int


@dataclass(frozen=True)
class GroupAnalysis:
    node_range: Tuple[int, int]
    nodes: List[gs.Node]
    entry_tensor: gs.Variable
    exit_tensor: gs.Variable
    main_input_indices: List[int]


@dataclass(frozen=True)
class StagePlan:
    input_ranges: List[HeightRange]
    output_ranges: List[HeightRange]
    conv_slices: Optional[List[ConvSlice]] = None
    conv_base_pads: Optional[List[int]] = None


@dataclass(frozen=True)
class GroupPlan:
    entry_ranges: List[HeightRange]
    stage_plans: List[StagePlan]
    output_ranges: List[HeightRange]


@dataclass
class TileBlock:
    """A scheduled work unit for one original op and one split tile."""

    orig_index: int
    tile_id: int
    node: gs.Node


@dataclass(frozen=True)
class TiledOpBuild:
    output_tiles: List[gs.Variable]
    nodes: List[gs.Node]
    blocks: List[TileBlock]


@dataclass(frozen=True)
class GroupBuild:
    nodes: List[gs.Node]
    blocks: List[TileBlock]
    concat_output: gs.Variable
    concat_node: gs.Node
