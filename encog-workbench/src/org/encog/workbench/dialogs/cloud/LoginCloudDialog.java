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

package org.encog.workbench.dialogs.cloud;

import java.awt.Frame;
import java.util.ArrayList;
import java.util.List;

import org.encog.workbench.dialogs.common.ComboBoxField;
import org.encog.workbench.dialogs.common.EncogPropertiesDialog;
import org.encog.workbench.dialogs.common.PasswordField;
import org.encog.workbench.dialogs.common.TextField;

public class LoginCloudDialog extends EncogPropertiesDialog {

	private TextField network;
	private TextField userID;
	private PasswordField password;
	
	public LoginCloudDialog(Frame owner) {
		super(owner);
		setTitle("Login to Encog Cloud");
		setSize(400,200);
		addProperty(this.network = new TextField("network","Network",true));
		addProperty(this.userID = new TextField("user id","User ID",true));
		addProperty(this.password = new PasswordField("password","Password",true));
		render();
	}

	public TextField getNetwork() {
		return network;
	}

	public TextField getUserID() {
		return userID;
	}

	public PasswordField getPassword() {
		return password;
	}
	
	

}
