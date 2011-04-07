package org.encog.app.analyst.util;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

public class CSVHeaders {
	
	private List<String> headerList = new ArrayList<String>();
	
	public CSVHeaders(File filename, boolean headers, CSVFormat format) {
		ReadCSV csv = null;
		try {
			csv = new ReadCSV(filename.toString(), headers, format);
			if (csv.next()) {
				if( headers ) {
					for( String str: csv.getColumnNames()) {
						headerList.add(str);
					}
				} else {
					for(int i=0;i<csv.getColumnCount();i++) {
						headerList.add("field:"+(i+1));
					}
				}
			}
		} finally {
			if (csv != null)
				csv.close();
		}
	}
	
	public CSVHeaders(String[] inputHeadings) {
		for(String header: inputHeadings) {
			this.headerList.add(header);
		}
	}

	public int size() {
		return this.headerList.size();
	}
	
	public List<String> getHeaders() {
		return this.headerList;
	}
	
	public String getHeader(int index) {
		return this.headerList.get(index);
	}
	
	public boolean isSeriesInput(int index) {
		String field = getHeader(index);
		return ( field.toLowerCase().startsWith("input:"));
	}
	
	public boolean isSeriesPredict(int index) {
		String field = getHeader(index);				
		return( field.toLowerCase().startsWith("predict:"));
	}
	
	public boolean isSeries(int index) {
		return isSeriesInput(index) || isSeriesPredict(index);
	}
	
	public String getBaseHeader(int index) { 
		String result = this.headerList.get(index);
		
		int loc = result.indexOf('(');
		if( loc!=-1 ) {
			result = result.substring(0,loc);
		}
		
		return result.trim();
	}
	
	public static String tagColumn(String name, int part, int timeSlice, boolean multiPart) {
		StringBuilder result = new StringBuilder();
		result.append(name);
		
		// is there any suffix?
		if( multiPart || timeSlice!=0 ) {
			result.append('(');
			
			// is there a part?
			if( multiPart ) {
				result.append('p');
				result.append(part);
			}
			
			// is there a timeslice?
			if( timeSlice!=0) {
				if( multiPart )
					result.append(',');
				result.append('t');
				if(timeSlice>0)
					result.append('+');
				result.append(timeSlice);
				
			}
			
			
			result.append(')');
		}
		return result.toString();
	}

}
