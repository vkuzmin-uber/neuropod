/* Copyright (c) 2020 UATC, LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package com.uber.neuropod;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.*;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class NeuropodTensorAllocatorTest {
    private NeuropodTensorAllocator allocator;
    private static final double EPSILON = 1E-6;

    @Before
    public void setUp() throws Exception {
        allocator = Neuropod.getGenericTensorAllocator();
    }

    @Test
    public void copyFromInt() {
        IntBuffer intBuffer = IntBuffer.allocate(4);
        for (int i=0; i<4; i++) {
            intBuffer.put(i);
        }
        intBuffer.rewind();
        long [] shape = new long[]{2,2};
        try(NeuropodTensor tensor = allocator.copyFrom(intBuffer, shape)) {
            assertArrayEquals(tensor.getDims(), shape);
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.INT32_TENSOR, tensor.getTensorType());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(intBuffer.get(i * 2 + j), tensor.getInt(i, j));
                }
            }
            IntBuffer outputBuffer = tensor.toIntBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(intBuffer.get(i), outputBuffer.get(i));
            }
        }
    }

    @Test
    public void copyFromLong() {
        LongBuffer longBuffer = LongBuffer.allocate(4);
        for (int i=0; i<4; i++) {
            longBuffer.put(i);
        }
        longBuffer.rewind();
        long [] shape = new long[]{2,2};
        try (NeuropodTensor tensor = allocator.copyFrom(longBuffer, shape)) {
            assertArrayEquals(tensor.getDims(), shape);
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.INT64_TENSOR, tensor.getTensorType());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(longBuffer.get(i * 2 + j), tensor.getLong(i, j));
                }
            }
            LongBuffer outputBuffer = tensor.toLongBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(longBuffer.get(i), outputBuffer.get(i));
            }
        }
    }

    @Test
    public void copyFromDouble() {
        DoubleBuffer doubleBuffer = DoubleBuffer.allocate(4);
        for (int i=0; i<4; i++) {
            doubleBuffer.put(i);
        }
        doubleBuffer.rewind();
        long [] shape = new long[]{2,2};
        try (NeuropodTensor tensor = allocator.copyFrom(doubleBuffer, shape)) {
            assertArrayEquals(tensor.getDims(), shape);
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.DOUBLE_TENSOR, tensor.getTensorType());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(doubleBuffer.get(i * 2 + j), tensor.getDouble(i, j), EPSILON);
                }
            }
            DoubleBuffer outputBuffer = tensor.toDoubleBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(doubleBuffer.get(i), outputBuffer.get(i), EPSILON);
            }

        }
    }

    @Test
    public void copyFromFloat() {
        FloatBuffer floatBuffer = FloatBuffer.allocate(4);
        for (int i=0; i<4; i++) {
            floatBuffer.put(i);
        }
        floatBuffer.rewind();
        long [] shape = new long[]{2,2};
        try (NeuropodTensor tensor = allocator.copyFrom(floatBuffer, shape)) {
            assertArrayEquals(tensor.getDims(), shape);
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.FLOAT_TENSOR, tensor.getTensorType());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(floatBuffer.get(i * 2 + j), tensor.getFloat(i, j), EPSILON);
                }
            }
            FloatBuffer outputBuffer = tensor.toFloatBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(floatBuffer.get(i), outputBuffer.get(i), EPSILON);
            }
        }
    }

    @Test
    public void copyFromString() {
        List<String> input = Arrays.asList("abc", "def", "ghi", "jkl");
        long [] shape = new long[]{2,2};
        try (NeuropodTensor tensor = allocator.copyFrom(input, shape)) {
            assertArrayEquals(tensor.getDims(), shape);
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.STRING_TENSOR, tensor.getTensorType());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(input.get(i * 2 + j), tensor.getString(i, j));
                }
            }
            List<String> output = tensor.toStringList();
            for (int i = 0; i < 4; i++) {
                assertEquals(input.get(i), output.get(i));
            }
        }

    }

    @Test
    public void tensorFromMemory() {
        TensorType type = TensorType.INT64_TENSOR;
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * type.getBytesPerElement()).order(ByteOrder.nativeOrder());
        LongBuffer longBuffer = buffer.asLongBuffer();
        for (int i = 0; i < 4; i++) {
            longBuffer.put(i);
        }

        long[] shape = new long[]{2, 2};
        try (NeuropodTensor tensor = allocator.tensorFromMemory(buffer, shape, type)) {
            assertArrayEquals(shape, tensor.getDims());
            assertEquals(4, tensor.getNumberOfElements());
            assertEquals(TensorType.INT64_TENSOR, tensor.getTensorType());
            for (int i=0; i<2; i++) {
                for (int j=0; j<2; j++) {
                    assertEquals(tensor.getLong(i, j), longBuffer.get(i*2 + j), EPSILON);
                }
            }
            LongBuffer outputBuffer = tensor.toLongBuffer();
            for (int i = 0; i < 4; i++) {
                assertEquals(longBuffer.get(i), outputBuffer.get(i));
            }
        }
    }

    @After
    public void tearDown() throws Exception {
        allocator.close();
    }
}
