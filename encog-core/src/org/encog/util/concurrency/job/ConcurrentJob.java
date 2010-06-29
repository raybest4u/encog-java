/*
 * Encog(tm) Core v2.5 
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008-2010 by Heaton Research Inc.
 * 
 * Released under the LGPL.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 * 
 * Encog and Heaton Research are Trademarks of Heaton Research, Inc.
 * For information on Heaton Research trademarks, visit:
 * 
 * http://www.heatonresearch.com/copyright.html
 */

package org.encog.util.concurrency.job;

import org.encog.EncogError;
import org.encog.StatusReportable;
import org.encog.util.concurrency.EncogConcurrency;
import org.encog.util.concurrency.TaskGroup;

/**
 * This class forms the basis for a job that can be run concurrently.
 */
public abstract class ConcurrentJob {

	/**
	 * The class to report status to.
	 */
	private final StatusReportable report;

	/**
	 * The number of tasks in this job.
	 */
	private int totalTasks;
	
	/**
	 * The current task.
	 */
	private int current;
	
	/**
	 * Flag to note that the job should stop.
	 */
	private boolean shouldStop = false;

	/**
	 * Construct a concurrent job.
	 * 
	 * @param report
	 *            The object to report status to.
	 */
	public ConcurrentJob(final StatusReportable report) {
		this.report = report;
		this.current = 1;
	}

	/**
	 * Load the subtasks.
	 * 
	 * @return The total number of subtasks.
	 */
	public abstract int loadWorkload();

	/**
	 * Perform one job unit.
	 * 
	 * @param context
	 *            The context for the job unit.
	 */
	public abstract void performJobUnit(JobUnitContext context);

	/**
	 * Process the job.
	 */
	public void process() throws EncogError {
		Object task;

		this.totalTasks = loadWorkload();
		int currentTask = 0;
		TaskGroup group = EncogConcurrency.getInstance().createTaskGroup();

		while( ((task = requestNextTask()) != null) && !shouldStop ) {
			currentTask++;
			final JobUnitContext context = new JobUnitContext();
			context.setJobUnit(task);
			context.setOwner(this);
			context.setTaskNumber(currentTask);

			final JobUnitWorker worker = new JobUnitWorker(context);
			EncogConcurrency.getInstance().processTask(worker, group);
		}

		group.waitForComplete();
		
		EncogConcurrency.getInstance().checkError();
	}

	/**
	 * Report the status for this job.
	 * 
	 * @param context
	 *            The job context.
	 * @param status
	 *            The status to report.
	 */
	public void reportStatus(final JobUnitContext context, final String status) {
		this.report.report(this.totalTasks, current++, status);
	}

	/**
	 * Request the next task to be processed.
	 * 
	 * @return The next task to be processed.
	 */
	public abstract Object requestNextTask();
	
	public boolean getShouldStop()
	{
		return this.shouldStop;
	}
	
	public void stop()
	{
		this.shouldStop = true;
	}
}
