/*
 * Encog Workbench v1.x
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008, Heaton Research Inc., and individual contributors.
 * See the copyright.txt in the distribution for a full listing of 
 * individual contributors.
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
 */
package org.encog.workbench.models;

import java.util.ArrayList;
import java.util.List;
import javax.swing.event.TableModelListener;
import javax.swing.table.TableModel;
import org.encog.matrix.Matrix;

public class MatrixTableModel implements TableModel {

	private final List<TableModelListener> listeners = new ArrayList<TableModelListener>();
	private final Matrix matrix;

	public MatrixTableModel(final Matrix matrix) {
		this.matrix = matrix;
	}

	public void addTableModelListener(final TableModelListener l) {
		this.listeners.add(l);
	}

	public Class<?> getColumnClass(final int columnIndex) {
		return String.class;
	}

	public int getColumnCount() {
		return this.matrix.getCols() + 1;
	}

	public String getColumnName(final int columnIndex) {
		if (columnIndex == 0) {
			return "";
		}
		return "Next Layer: " + columnIndex;
	}

	public int getRowCount() {
		return this.matrix.getRows();
	}

	public Object getValueAt(final int rowIndex, final int columnIndex) {
		if (columnIndex == 0) {
			if (rowIndex == this.matrix.getRows() - 1) {
				return "Threshold: ";
			}
			return "This Layer: " + (rowIndex + 1);

		}
		return "" + this.matrix.get(rowIndex, columnIndex - 1);
	}

	public boolean isCellEditable(final int rowIndex, final int columnIndex) {
		if (columnIndex == 0) {
			return false;
		}
		return true;

	}

	public void removeTableModelListener(final TableModelListener l) {
		this.listeners.remove(l);
	}

	public void setValueAt(final Object value, final int rowIndex,
			final int columnIndex) {
		this.matrix.set(rowIndex, columnIndex - 1, Double
				.parseDouble((String) value));
	}
}
