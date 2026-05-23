"""Protocol grammar — the COSMIC ASCEC workflow stage parser.

A ``.asc`` file may carry an embedded *protocol* line that chains several
workflow stages, e.g.::

    .asc, r3, opt --redo=3 preopt.inp launcher.sh, cosmic --th=2

The grammar:

* Stages are separated by ``,`` or the literal word ``then``. A separator may
  be attached to a token (``r3,``) or stand alone (``r3 , opt``).
* A trailing ``.`` on a stage (or a standalone ``.``) marks *pause after this
  stage* — the workflow stops for manual review.

Stage tokens:

==================  ============  ============================================
Token               Stage type    What it does
==================  ============  ============================================
``rN``              replication   Run N annealing replicas in parallel.
``opt`` / ``optimization``        Per-structure geometry optimization.
``cosmic``          cosmic        Cluster optimized structures into motifs.
``ref`` / ``refinement``          Refine motifs at a higher level of theory.
``eref`` / ``energy_refinement``  Energy-only refinement (composite Gibbs).
==================  ============  ============================================

Stages are returned as loosely-typed dicts (``{'type': ..., 'args': [...],
'pause_after': bool, ...}``) rather than typed objects so the per-stage
runners in :mod:`~cosmic_ascec.workflow.stages` can index them directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    "contains_workflow_separator",
    "parse_workflow_stages",
    "finalize_stage",
]


def contains_workflow_separator(args: List[str]) -> bool:
    """Check if command line arguments contain ',' or 'then' separator (workflow mode)."""
    for arg in args:
        if not arg:
            continue
        token = arg.strip().lower()
        if token == 'then' or token == ',':
            return True
        # Support comma-attached forms such as "r3," or "input.asc,"
        if ',' in token:
            return True
    return False


def parse_workflow_stages(args: List[str]) -> List[Dict[str, Any]]:
    """
    Parse compound workflow commands with ',' or 'then' separator.
    Supports both space-separated and comma-attached formats.

    Examples:
        ascec04 at_annealing.in , r3 , opt --redo=3 preopt.inp launcher.sh , cosmic --th=2
        ascec04 at_annealing.asc, r3, opt --redo=3 preopt.inp launcher.sh, cosmic --th=2
        ascec04 .asc, r3, opt -c --redo=3 preopt.inp launcher.sh, cosmic --th=2

        With pause after stage (using dot separator):
        ascec04 .asc, r3, opt --redo=3 preopt.inp launcher.sh. cosmic --th=2
            (dot after launcher.sh means pause after optimization stage for manual review)

        With auto-selection flags:
        ascec04 at_annealing.asc , r3 , opt -a --redo=3 preopt.inp launcher.sh
            (-a: Process all result_*.xyz files separately)
        ascec04 at_annealing.asc , r3 , opt -c --redo=3 preopt.inp launcher.sh
            (-c: Combine all result_*.xyz into combined_r{N}.xyz first, then process)

        Flag meanings:
            --redo=N: Redo entire stage (optimization+cosmic or refinement+cosmic) up to N times

        Note: Launch failures (instant crashes) are automatically retried up to 10 times.
        Once an optimization run starts running normally, it will not be retried regardless of exit code.

    Returns:
        List of stage dictionaries with 'type', 'args', and optional 'pause_after' keys
    """
    stages = []
    current_stage = []
    pause_after_current = False

    for arg in args:
        # Skip blank/empty arguments (allows blank lines in protocol)
        if not arg or not arg.strip():
            continue

        # Handle comma or dot attached to argument (e.g., "r3," or "launcher.sh.")
        has_comma = ',' in arg and arg != ','
        has_dot = '.' in arg and not arg.startswith('.') and not arg.endswith(('.inp', '.sh', '.xyz', '.in'))

        if has_comma or (has_dot and len(arg) > 1 and arg[-1] == '.'):
            # Split by separator (comma or dot)
            if has_dot and arg[-1] == '.':
                # Dot at end means pause after this stage
                part = arg[:-1].strip()  # Remove the dot
                if part:
                    current_stage.append(part)
                # Finalize stage with pause marker
                if current_stage:
                    stage = finalize_stage(current_stage, pause_after=True)
                    if stage:
                        stages.append(stage)
                    current_stage = []
                    pause_after_current = False
            else:
                # Split by comma
                parts = arg.split(',')
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part:
                        current_stage.append(part)
                    # After each part except the last, finalize stage
                    if i < len(parts) - 1:
                        if current_stage:
                            stage = finalize_stage(current_stage, pause_after=pause_after_current)
                            if stage:
                                stages.append(stage)
                            current_stage = []
                            pause_after_current = False
        elif arg in [',', 'then']:
            # Separator found - finalize current stage if it has content
            if current_stage:
                stage = finalize_stage(current_stage, pause_after=pause_after_current)
                if stage:
                    stages.append(stage)
                current_stage = []
                pause_after_current = False
        elif arg == '.':
            # Standalone dot - mark current stage for pause
            pause_after_current = True
        else:
            # Regular argument - add to current stage
            current_stage.append(arg)

    # Don't forget the last stage
    if current_stage:
        stage = finalize_stage(current_stage, pause_after=pause_after_current)
        if stage:
            stages.append(stage)

    return stages


def finalize_stage(stage_args: List[str], pause_after: bool = False) -> Optional[Dict[str, Any]]:
    """
    Convert raw stage arguments into structured stage dictionary.

    Args:
        stage_args: List of arguments between ',' or 'then' separators
        pause_after: If True, workflow should pause after this stage completes

    Returns:
        Dictionary with 'type', 'args', and optional 'pause_after' keys, or None if invalid
    """
    if not stage_args:
        return None

    first_arg = stage_args[0].lower()

    # Replication stage: r3, r5, etc.
    if first_arg.startswith('r') and first_arg[1:].isdigit():
        stage_dict = {
            'type': 'replication',
            'num_replicas': int(first_arg[1:]),
            'args': stage_args[1:]
        }
        if pause_after:
            stage_dict['pause_after'] = True
        return stage_dict

    # Optimization stage: opt/optimization ...
    # New naming: "opt" means Optimization stage (internal type: optimization)
    elif first_arg in ['opt', 'optimization']:
        stage_dict = {
            'type': 'optimization',
            'args': stage_args[1:]
        }
        if pause_after:
            stage_dict['pause_after'] = True
        return stage_dict

    # cosmic stage: cosmic ...
    elif first_arg in ['cosmic', 'cosmic']:
        stage_dict = {
            'type': 'cosmic',
            'args': stage_args[1:]
        }
        if pause_after:
            stage_dict['pause_after'] = True
        return stage_dict

    # Refinement stage: ref/refinement ... (internal type: refinement)
    elif first_arg in ['ref', 'refinement']:
        # Parse opt stage: opt [flags] template_file launcher_file
        # Find template and launcher (non-flag arguments)
        remaining_args = stage_args[1:]
        flags = []
        files = []

        for arg in remaining_args:
            if arg.startswith('--') or arg.startswith('-'):
                flags.append(arg)
            else:
                files.append(arg)

        # Expect at least 2 files: template and launcher
        template_inp = files[0] if len(files) >= 1 else None
        launcher_sh = files[1] if len(files) >= 2 else None

        stage_dict = {
            'type': 'refinement',
            'args': flags,  # Only flags, not file paths
            'template_inp': template_inp,
            'launcher_sh': launcher_sh
        }
        if pause_after:
            stage_dict['pause_after'] = True
        return stage_dict

    # Energy refinement stage: eref/energy_refinement ... (internal type: energy_refinement)
    elif first_arg in ['eref', 'energy_refinement']:
        remaining_args = stage_args[1:]
        flags = []
        files = []

        for arg in remaining_args:
            if arg.startswith('--') or arg.startswith('-'):
                flags.append(arg)
            else:
                files.append(arg)

        template_inp = files[0] if len(files) >= 1 else None
        launcher_sh = files[1] if len(files) >= 2 else None

        stage_dict = {
            'type': 'energy_refinement',
            'args': flags,
            'template_inp': template_inp,
            'launcher_sh': launcher_sh
        }
        if pause_after:
            stage_dict['pause_after'] = True
        return stage_dict

    else:
        print(f"Warning: Unknown stage type '{first_arg}', skipping.")
        return None
