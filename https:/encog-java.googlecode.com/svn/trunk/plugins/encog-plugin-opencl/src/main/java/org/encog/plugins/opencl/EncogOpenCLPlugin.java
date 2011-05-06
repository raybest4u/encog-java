package org.encog.plugins.opencl;

import java.nio.FloatBuffer;

import org.encog.util.file.ResourceInputStream;

import com.nativelibs4java.opencl.CLBuildException;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLFloatBuffer;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.util.NIOUtils;

public class EncogOpenCLPlugin {
	public static void main(String[] args) {

		CLContext context = JavaCL.createBestContext();
		CLQueue queue = context.createDefaultQueue();

				
		String myKernelSource = ResourceInputStream.readResourceAsString("org/encog/plugins/opencl/kernels/simple.cl");
		
		FloatBuffer array1 = NIOUtils.directFloats(10, context.getByteOrder());
		FloatBuffer array2 = NIOUtils.directFloats(10, context.getByteOrder());
		FloatBuffer resultArray = NIOUtils.directFloats(10, context.getByteOrder());
		
		CLFloatBuffer b1 = context.createFloatBuffer(Usage.Input, array1, true);
		CLFloatBuffer b2 = context.createFloatBuffer(Usage.Input, array1, true);
		CLFloatBuffer b3 = context.createFloatBuffer(Usage.Output, resultArray, false);

		CLProgram program;
		try {
			program = context.createProgram(myKernelSource).build();
			CLKernel kernel = program.createKernel(
			        "simpleKernel", 
			        b1,
			        b2,
			        b3
			);
			
			
			
			CLEvent kernelCompletion;
			// The same kernel can be safely used by different threads, as long as setArgs + enqueueNDRange are in a synchronized block
			synchronized (kernel) {
			    // setArgs will throw an exception at runtime if the types / sizes of the arguments are incorrect
			    kernel.setArgs(b1,b2,b3);

			   // Ask for 1-dimensional execution of length dataSize, with auto choice of local workgroup size :
			    kernelCompletion = kernel.enqueueNDRange(queue, new int[] { 10 }, null);
			}
			kernelCompletion.waitFor(); // better not to wait for it but to pass it as a dependent event to some other queuable operation (CLBuffer.read, for instance)
			
			
			
		} catch (CLBuildException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		
	}
}