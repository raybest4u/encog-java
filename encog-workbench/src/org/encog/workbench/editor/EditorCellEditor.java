/*
 * Encog(tm) Workbench v2.4
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

package org.encog.workbench.editor;

import java.awt.Component;
import java.util.ArrayList;
import java.util.EventObject;
import java.util.List;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.event.CellEditorListener;
import javax.swing.table.TableCellEditor;

public class EditorCellEditor implements TableCellEditor {
	
	private List<CellEditorListener> listeners = new ArrayList<CellEditorListener>();
	private JTextField editorText;
	private JComboBox editorBoolean;
	private Class currentClass;
	private Object value;
	private int row;
	private int column;
	private JTable table;
	
	public static final String[] BOOLEAN_VALUES = {"true","false"};
	
	public EditorCellEditor()
	{
		this.editorText = new JTextField();
		this.editorBoolean = new JComboBox(BOOLEAN_VALUES);
	}
	
	public Component getTableCellEditorComponent(JTable table, Object value,
			boolean isSelected, int row, int column) {
		this.row = row;
		this.column = column;
		this.table = table;
		this.currentClass = value.getClass();
		this.value = value;
		if( this.currentClass == Boolean.class )
		{
			this.editorBoolean.setVisible(true);
			if( (Boolean)value == true )
				this.editorBoolean.setSelectedIndex(0);
			else
				this.editorBoolean.setSelectedIndex(1);
			
			return this.editorBoolean;	
		}
		else
		{
			this.editorText.setVisible(true);
			this.editorText.setText(value.toString());
			return this.editorText;
		}
	}

	public void addCellEditorListener(CellEditorListener l) {
		this.listeners.add(l);
		
	}

	public void cancelCellEditing() {
		if( this.currentClass == Boolean.class ) {
			this.editorBoolean.setVisible(false);
			this.editorBoolean.setSelectedIndex(0);
		}
		else
		{
			this.editorText.setVisible(false);
			this.editorText.setText("");
		}
		
	}

	public Object getCellEditorValue() {
		stopCellEditing();
		return this.value;
	}

	public boolean isCellEditable(EventObject anEvent) {
		return true;
	}

	public void removeCellEditorListener(CellEditorListener l) {
		this.listeners.remove(l);
		
	}

	public boolean shouldSelectCell(EventObject anEvent) {
		return true;
	}

	public boolean stopCellEditing() {
		if( this.currentClass == Boolean.class ) {
			this.value = this.editorBoolean.getSelectedIndex()==0;
			System.out.println(this.value);
			this.editorBoolean.setVisible(false);
		}
		else
		{
			this.editorText.setVisible(false);
			this.value = this.editorText.getText();
		}
		
		if( this.table!=null)
		{
			this.table.setValueAt(this.value, this.row, this.column);
		}
		return true;
	}
	
	public void clearSelection()
	{
		this.editorBoolean.setVisible(false);
		this.editorText.setVisible(false);
	}

}
