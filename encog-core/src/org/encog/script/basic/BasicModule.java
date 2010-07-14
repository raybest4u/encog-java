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

package org.encog.script.basic;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.encog.EncogError;
import org.encog.script.EncogScript;
import org.encog.script.basic.error.BasicError;
import org.encog.script.basic.error.ErrorNumbers;
import org.encog.script.basic.util.BasicUtil;
import org.encog.util.ReflectionUtil;

/**
 * A module is a single Encog resource, it is made up of lines.
 */
public class BasicModule  {
	
	private Map<String,BasicLine> programLabels = new HashMap<String,BasicLine>();
	private Map<String,BasicLine> subLabels = new HashMap<String,BasicLine>();
	private List<BasicLine> programLines = new ArrayList<BasicLine>();
	private List<BasicLine> addto;
	private BasicProgram program;
	
	public BasicModule(BasicProgram program)
	{
		this.program = program;
	}

	public void clear()
	{
		programLines.clear();
		programLabels.clear();
	}
	
	public void load(EncogScript script) {
		try {
			this.addto = this.programLines;
			
			final InputStream is = new ByteArrayInputStream(script.getSource()
					.getBytes());
			final BufferedReader reader = new BufferedReader(
					new InputStreamReader(is));
			String line;

			while ((line = reader.readLine()) != null) {
				String check = line.trim();
				
				if( check.length()==0 )
					continue;
				
				if( check.charAt(0)=='\'')
					continue;
				
				addLine(line);
			}
		} catch (IOException e) {
			throw new EncogError(e);
		}
	}
		
	public void addLine(String line)
	{
		boolean createSub = false;
		int ptr;
		int ptr2;
		String label = "";
		String subLabel = null;
		BasicLine bl;

			line = BasicUtil.basicToUpper(line);

			// Look for a label

			ptr=line.indexOf(':');
			
			if(ptr!=-1)
			{
				// Check to see if this ':' is really that of a label
				ptr2=0;
				while((Character.isLetterOrDigit(line.charAt(ptr2))) && (line.charAt(ptr2)!=32) )
					ptr2++;

				// If it is a label, clip it and mark it as a label
				if(ptr==ptr2)
				{
					label = line.substring(0,ptr);
					line = line.substring(ptr+1);
				}
			}

			// Not a label, so check for a FUNCTION or SUB

			if(label.length()==0)
			{
				int l = 0;
				
				while( ( (line.charAt(l)==' ') || (line.charAt(l)=='\t') ) && l<line.length() )
					l++;
				
				ptr=BasicUtil.findKeyword(line,"SUB");
				if(ptr==-1)
					ptr=BasicUtil.findKeyword(line,"FUNCTION");

				if(ptr!=-1)
				{
					ptr = line.indexOf(' ',ptr);
					
					if(ptr!=-1)
					{
						while( " \t".indexOf(line.charAt(ptr))!=-1 )
							ptr++;
						
						StringBuilder b = new StringBuilder();
						
						while( ptr<line.length() && " \t(".indexOf(line.charAt(ptr))==-1  )
							b.append(line.charAt(ptr++));
						subLabel = b.toString();
					}
					
					if( BasicUtil.findKeyword(subLabel)!=null)
						throw(new BasicError(ErrorNumbers.errorIllegalFunctionName));

					if(this.programLabels.containsKey(label))
						throw(new BasicError(ErrorNumbers.errorAlreadyDeclared));
					
					createSub = true;
				}
			}

			if(label.length()>0)
			{
				if(BasicUtil.findKeyword(label)!=null)
					throw( new BasicError(ErrorNumbers.errorIllegalFunctionName));
			}

			int number = this.addto.size();
			this.addto.add(bl=new BasicLine(line));
			bl.setNumber(number);
			if( label.length()>0 )
			{
				this.programLabels.put(label,bl);			
			}	
			
			if( subLabel!=null )
			{
				this.subLabels.put(subLabel, bl);
			}
			
			if( createSub )
				this.addto = bl.getSub();
	}
	
	public BasicLine findFunction(String label)
	{
		String key = label.toUpperCase();
		return this.subLabels.get(key);
	}	
	
	public List<BasicLine> getProgramLines() {
		return programLines;
	}
	
	public Map<String,BasicLine> getProgramLabels()
	{
		return this.programLabels;
	}
	
	public BasicProgram getProgram() {
		return program;
	}
}
