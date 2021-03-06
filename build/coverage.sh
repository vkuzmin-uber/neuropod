#!/bin/bash
set -e

# Use the virtualenv
source .neuropod_venv/bin/activate

BASE_PATH=`pwd`

# Merge all the python coverage reports and print
pushd source/python
coverage combine
coverage report

# Generate an xml report and fix paths
coverage xml
sed -i "s+$BASE_PATH/++g" coverage.xml
sed -i 's+name="\..*neuropod\.python+name="neuropod+g' coverage.xml

popd

pushd source

# Merge all the coverage reports for native code
./external/llvm_toolchain/bin/llvm-profdata merge -output=/tmp/neuropod_coverage/code.profdata /tmp/neuropod_coverage/code-*.profraw

# Generate a coverage report and fix paths
bazel query 'kind("cc_binary|cc_test", ...)' | sed 's/\/\//-object bazel-bin\//g' |  sed 's/:/\//g' | paste -sd ' ' | xargs ./bazel-source/external/llvm_toolchain/bin/llvm-cov export --format=lcov -instr-profile=/tmp/neuropod_coverage/code.profdata -ignore-filename-regex="(external|tests)" > native_coverage.txt
sed -i 's+/proc/self/cwd/+source/+g' native_coverage.txt

# Get Java coverage and fix paths
cp -f ./bazel-source/bazel-out/_coverage/_coverage_report.dat java_coverage.txt
sed -i 's+SF:neuropod/bindings+SF:source/neuropod/bindings+g' java_coverage.txt

# Combine into one report
lcov -a java_coverage.txt -a native_coverage.txt -o coverage.txt
rm java_coverage.txt native_coverage.txt

# Display coverage information
lcov -l coverage.txt
popd
