/*
 * Encog Artificial Intelligence Framework v2.x
 * Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008-2009, Heaton Research Inc., and individual contributors.
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
package org.encog.parse;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.encog.parse.recognize.Recognize;
import org.encog.parse.signal.Signal;
import org.encog.parse.units.UnitManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Parse {  
  private static UnitManager unitManager;
  private ParseTemplate template;
  public static final String RESOURCE_NAME = "org/encog/data/template.eg";
  public static final String RESOURCE_ITEM_NAME = "parse-native";
  
  /**
	 * The logging object.
	 */
	@SuppressWarnings("unused")
	final private Logger logger = LoggerFactory.getLogger(this.getClass());

  public Signal parseFile(String name)
  {
	  try
	  {
    FileReader fileReader = new FileReader(name);
    BufferedReader bufferedReader = new BufferedReader(fileReader);

    String contents="";
    String line="";

    while ( (line=bufferedReader.readLine())!=null )
      contents+=line+"\r\n";

    bufferedReader.close();
    fileReader.close();
    return parse(contents);
	  }
	  catch(IOException e)
	  {
		  if( logger.isErrorEnabled())
	    	{
	    		logger.error("Exception",e);
	    	}
	    	throw new ParseError(e);
	  }
  }

  public static void setUnitMananger(UnitManager unitManager)
  {
    Parse.unitManager = unitManager;
  }

  public static UnitManager getUnitManager()
  {
    return Parse.unitManager;
  }

  private boolean parseIteration(Signal input)
  {
    boolean changed = false;
    
    if( this.template == null )
    {
    	throw new ParseError("Must load a template before calling the Parse object.");
    }

    for(Recognize recognize:template.getRecognizers()) {
      if(recognize.recognize(input))
        changed = true;
    }
    return changed;
  }

  public Signal parse(String input)
  {
    Signal result = new Signal(input+" ");
    result.resetDelta();

    do
    {
      result.resetDelta();
      parseIteration(result);
    }
    while( result.getDelta() );
    return result;
  }

/**
 * @return the template
 */
public ParseTemplate getTemplate() {
	return template;
}

public void load()  {
	//EncogPersistedCollection encog = new EncogPersistedCollection();
	//encog.loadResource(Parse.RESOURCE_NAME);
	//this.template = (ParseTemplate) encog.find(Parse.RESOURCE_ITEM_NAME);
}




}
