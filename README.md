# AutoStrategizer
The AutoStrategizer is an automated framework that utilizes complex hardware links while preserving the simplified abstraction level for the user.

The operations we support include moving, distribution, and consolidation of memory across the node.
For each of them, our AutoStrategizer framework proposes a task graph that transparently improves performance, in terms of latency or bandwidth, compared to naive strategies.

## Clone GitHub pository
`git clone git@github.com:Darptolus/AutoStrategyzer.git`

## Compile and Run Tests

### Compile
`sh runme.sh`

### Run
`cd build/tests`

`./main_auto`

## Integration with LLVM

For our evaluation, we integrated the AutoStrategizer as a C++ library into the LLVM-OpenMP runtime infrastructure.

[LLVM Strategizer](https://github.com/rodrigo-ceccato/llvm-strategizer)
