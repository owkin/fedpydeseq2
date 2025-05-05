# Workflow graph

The workflow graph below illustrates the sequence of operations in the design matrix construction process. It shows how data flows between local centers and the aggregation server during the `compute_cook_distance` function.

![Workflow Graph](workflow_graph.png)

For a detailed breakdown of the shared states and their contents at each step, please refer to the table below.

# API

::: fedpydeseq2.core.deseq2_core.compute_cook_distance
    options:
        show_submodules: true


# Table with shared quantities between centers and server

{% include "api/core/deseq2_core/compute_cook_distance/shared.md" %}
