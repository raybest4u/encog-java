package org.encog.plugins.opencl;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import org.encog.EncogError;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.plugin.EncogPluginType1;
import org.encog.util.EngineArray;
import org.encog.util.file.ResourceInputStream;

import com.nativelibs4java.opencl.CLBuildException;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLFloatBuffer;
import com.nativelibs4java.opencl.CLIntBuffer;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.util.NIOUtils;


public class EncogOpenCLPlugin implements EncogPluginType1 {
	
	public static final int PARAM_COUNT = 5;
	
	private CLContext context;
	private CLQueue queue;
	private CLProgram program;
	private CLKernel kernel;
	private IntBuffer paramBuffer;
	private CLIntBuffer paramCLBuffer;
	private ExpandingBuffer weightsBuffer;
	private ExpandingBuffer layerOutputBuffer;
	private ExpandingBuffer outputBuffer;

	public EncogOpenCLPlugin() {
				
		this.context = JavaCL.createBestContext();
		this.queue = context.createDefaultQueue();
		
		this.paramBuffer = NIOUtils.directInts(PARAM_COUNT, context.getByteOrder());
		this.paramCLBuffer = context.createIntBuffer(Usage.Input, paramBuffer, false);
		
		this.weightsBuffer = new ExpandingBuffer(context,false);
		this.layerOutputBuffer = new ExpandingBuffer(context,false);
		this.outputBuffer = new ExpandingBuffer(context,true);

		String calcLayerFloat = ResourceInputStream.readResourceAsString("org/encog/plugins/opencl/kernels/calcLayerFloat.cl");
		
		try {
			this.program = context.createProgram(calcLayerFloat).build();
			this.kernel = program.createKernel("calculateLayer");
		} catch (CLBuildException e) {
			throw new EncogError(e);
		}
	}

	@Override
	public int getPluginType() {
		return 1;
	}

	@Override
	public int getPluginServiceType() {
		return EncogPluginType1.SERVICE_TYPE_CALCULATION;
	}

	@Override
	public String getPluginName() {
		return "hri-plugin-opencl";
	}

	@Override
	public String getPluginDescription() {
		return "The Heaton Research OpenCL plugin.  Allows faster execution with a GPU.";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public final void calculateGradient(final double[] gradients,
			final double[] layerOutput, final double[] weights,
			final double[] layerDelta, final ActivationFunction af,
			final int index, final int fromLayerIndex, final int fromLayerSize,
			final int toLayerIndex, final int toLayerSize) {
		int yi = fromLayerIndex;
		for (int y = 0; y < fromLayerSize; y++) {
			final double output = layerOutput[yi];
			double sum = 0;
			int xi = toLayerIndex;
			int wi = index + y;
			for (int x = 0; x < toLayerSize; x++) {
				gradients[wi] += output * layerDelta[xi];
				sum += weights[wi] * layerDelta[xi];
				wi += fromLayerSize;
				xi++;
			}

			layerDelta[yi] = sum * af.derivativeFunction(layerOutput[yi]);
			yi++;
		}
	}	
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public final int calculateLayer(final double[] weights,
			final double[] layerOutput, final int startIndex,
			final int outputIndex, final int outputSize, final int inputIndex,
			final int inputSize) {
		
		int[] paramArray = {startIndex,outputIndex,outputSize,inputIndex,inputSize};
		
		// create NIO buffers		
		this.weightsBuffer.setSize(weights.length);
		this.layerOutputBuffer.setSize(layerOutput.length);
		this.outputBuffer.setSize(outputSize);
		
		paramBuffer.rewind();
		paramBuffer.put(paramArray);
		
		this.weightsBuffer.set(weights);
		this.layerOutputBuffer.set(layerOutput);		
				
		// execute
		CLEvent kernelCompletion = null;
		// The same kernel can be safely used by different threads, as long as setArgs + enqueueNDRange are in a synchronized block
		synchronized (kernel) {
		    kernel.setArgs(this.weightsBuffer.getCLBuffer(),paramCLBuffer,this.layerOutputBuffer.getCLBuffer(),this.outputBuffer.getCLBuffer());
		    kernelCompletion = kernel.enqueueNDRange(queue, new int[] { outputSize }, new int[] { 1 } );
		}
		//kernelCompletion.waitFor(); // better not to wait for it but to pass it as a dependent event to some other queuable operation (CLBuffer.read, for instance)
		
		FloatBuffer f = this.outputBuffer.getCLBuffer().read(queue, kernelCompletion);
		for(int i=0;i<outputSize;i++) {
			layerOutput[outputIndex+i] = f.get(i);
		}
		
		if( kernelCompletion!=null )
			kernelCompletion.release();
		
		return startIndex+(inputSize*outputSize);
	}

	@Override
	public int getLogLevel() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void log(int level, String message) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void log(int level, Throwable t) {
		// TODO Auto-generated method stub
		
	}
	
}