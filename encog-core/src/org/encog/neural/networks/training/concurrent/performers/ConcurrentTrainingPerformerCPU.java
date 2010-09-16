package org.encog.neural.networks.training.concurrent.performers;

import java.util.concurrent.atomic.AtomicBoolean;

import org.encog.engine.opencl.EncogCLDevice;
import org.encog.neural.NeuralNetworkError;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.concurrent.jobs.TrainingJob;

public class ConcurrentTrainingPerformerCPU implements
		ConcurrentTrainingPerformer, Runnable {

	private AtomicBoolean ready = new AtomicBoolean(true);
	private TrainingJob currentJob;

	@Override
	public void perform(TrainingJob job) {
		if (this.ready.get() == false) {
			throw new NeuralNetworkError(
					"Performer is already performing a job.");
		}

		setupJob(job);

		this.ready.set(false);
		this.currentJob = job;

		Thread t = new Thread(this);
		t.start();
	}

	protected void setupJob(TrainingJob job) {
		// nothing to be done
	}

	@Override
	public boolean ready() {
		return ready.get();
	}

	public void run() {
		try {
			EncogCLDevice device = null;
			if (this instanceof ConcurrentTrainingPerformerOpenCL) {
				device = ((ConcurrentTrainingPerformerOpenCL) this).getDevice();
			}
			this.currentJob.createTrainer(device);
			Train train = this.currentJob.getTrain();
			int interation = 1;

			while (currentJob.shouldContinue()) {
				train.iteration();
				interation++;
			}
		} catch (Throwable t) {
			currentJob.setError(t);
		} finally {
			this.ready.set(true);
		}
	}
}
