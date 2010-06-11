/*
 * Encog(tm) Workbench v2.5
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

package org.encog.workbench.dialogs.common;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;

public class PasswordField extends PropertiesField {
	private String value;
	
	public PasswordField(String name, String label, boolean required) {
		super(name, label, required);
	}

	public String getValue() {
		return value;
	}

	public void setValue(String value) {
		this.value = value;
		((JTextField)this.getField()).setText(value);
	}

	@Override
	public void collect() throws ValidationException {
		this.value = ((JTextField)this.getField()).getText();
		if( this.value.length()<1 && this.isRequired())
		{
			throw new ValidationException("The field " + this.getName() + " is required.");
		}
	}
	
	public int createField(JPanel panel, int x, int y,int width)
	{
		JPasswordField field = new JPasswordField();
		
		this.setField(field);
		field.setLocation(x, y);
		field.setSize(field.getPreferredSize());
		field.setSize(width,field.getHeight());
		JLabel label = createLabel();
		label.setLocation(label.getX(), y);
		panel.add(label);
		panel.add(field);
		
		return y+field.getHeight();
	}
	

}
