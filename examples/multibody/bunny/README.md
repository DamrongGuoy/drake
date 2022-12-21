# Deformable bunny

This is an example of simulation of deformable bodies in Drake.
The example drops a deformable bunny on the ground to test the bunny mesh that
doesn't have zero gradient distance field.

## Run visualizer

```
bazel run //tools:meldis -- --open-window &
```

## Run the example

```
bazel run --config=omp --copt=-march=native //examples/multibody/bunny:bunny
```

## Options

There are a few command-line options that you can use to adjust the physical
properties of the deformable body. Use `--help` to see the list.
