package org.encog.util.cl.kernels;

import static org.jocl.CL.clSetKernelArg;

import org.encog.util.cl.EncogCLDevice;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_context;
import org.jocl.cl_mem;

/**
 * A very simple kernel, used to add a vector. Not actually used by Encog, it is
 * a simple test case to verify that OpenCL is working.
 */
public class KernelVectorAdd extends EncogKernel {

	/**
	 * Construct a simple kernel to add two vectors.
	 * 
	 * @param context
	 *            The context to use.
	 */
	public KernelVectorAdd(final cl_context context) {
		super(context, "org/encog/data/KernelVectorAdd.txt", "VectorAdd");
	}

	/**
	 * Perform the addition.
	 * 
	 * @param device
	 *            The OpenCL device to use.
	 * @param inputA
	 *            The first vector to add.
	 * @param inputB
	 *            The second vector to add.
	 * @return The result of the addition.
	 */
	public double[] add(final EncogCLDevice device, final double[] inputA,
			final double[] inputB) {
		// Create input- and output data
		final int n = inputA.length;
		final float[] srcArrayA = new float[n];
		final float[] srcArrayB = new float[n];
		final float[] dstArray = new float[n];

		for (int i = 0; i < n; i++) {
			srcArrayA[i] = (float) inputA[i];
			srcArrayB[i] = (float) inputB[i];
		}
		final Pointer srcA = Pointer.to(srcArrayA);
		final Pointer srcB = Pointer.to(srcArrayB);
		final Pointer dst = Pointer.to(dstArray);

		// Allocate the memory objects for the input- and output data
		final cl_mem[] memObjects = new cl_mem[3];
		memObjects[0] = CL.clCreateBuffer(getContext(), CL.CL_MEM_READ_ONLY
				| CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * n, srcA, null);
		memObjects[1] = CL.clCreateBuffer(getContext(), CL.CL_MEM_READ_ONLY
				| CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * n, srcB, null);
		memObjects[2] = CL.clCreateBuffer(getContext(), CL.CL_MEM_READ_WRITE,
				Sizeof.cl_float * n, null, null);
		
        // Set the arguments for the kernel
        clSetKernelArg(getKernel(), 0, 
            Sizeof.cl_mem, Pointer.to(memObjects[0]));
        clSetKernelArg(getKernel(), 1, 
            Sizeof.cl_mem, Pointer.to(memObjects[1]));
        clSetKernelArg(getKernel(), 2, 
            Sizeof.cl_mem, Pointer.to(memObjects[2]));

		// Set the work-item dimensions
		final long[] global_work_size = new long[] { n };
		final long[] local_work_size = new long[] { 1 };

		// Execute the kernel
		CL.clEnqueueNDRangeKernel(device.getCommands(), getKernel(), 1, null,
				global_work_size, local_work_size, 0, null, null);

		// Read the output data
		CL.clEnqueueReadBuffer(device.getCommands(), memObjects[2], CL.CL_TRUE,
				0, n * Sizeof.cl_float, dst, 0, null, null);

		final double[] result = new double[dstArray.length];

		for (int i = 0; i < dstArray.length; i++) {
			result[i] = dstArray[i];
		}

		return result;
	}

}
