package org.encog.app.analyst.script.prop;

import org.encog.app.analyst.AnalystError;
import org.encog.app.analyst.util.ConvertStringConst;

/**
 * A property entry for the Encog Analyst. Properties have a name and section.
 * 
 */
public class PropertyEntry implements Comparable<PropertyEntry> {

	private final PropertyType entryType;
	private final String name;
	private final String section;

	public PropertyEntry(PropertyType entryType, String name, String section) {
		super();
		this.entryType = entryType;
		this.name = name;
		this.section = section;
	}

	/**
	 * @return the entryType
	 */
	public PropertyType getEntryType() {
		return entryType;
	}

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @return the section
	 */
	public String getSection() {
		return section;
	}

	@Override
	public int compareTo(PropertyEntry o) {
		return this.name.compareTo(o.name);
	}

	public String getKey() {
		return section + "_" + name;
	}

	/** {@inheritDoc} */
	public String toString() {
		StringBuilder result = new StringBuilder("[");
		result.append(getClass().getSimpleName());
		result.append(" name=");
		result.append(this.name);
		result.append(", section=");
		result.append(this.section);
		result.append("]");
		return result.toString();
	}

	public static String dotForm(String section, String subSection, String name) {
		StringBuilder result = new StringBuilder();
		result.append(section);
		result.append('.');
		result.append(subSection);
		result.append('.');
		result.append(name);
		return result.toString();
	}

	public void validate(String section, String subSection, String name,
			String value) {
		if( value==null || value.length()==0 )
			return;
		
		try {
			switch (getEntryType()) {
			case TypeBoolean:
				if( Character.toUpperCase(value.charAt(0))!='T' && Character.toUpperCase(value.charAt(0))!='F' )
				{
					StringBuilder result = new StringBuilder();
					result.append("Illegal boolean for ");
					result.append(dotForm(section,subSection,name));
					result.append(", value is ");
					result.append(value);
					result.append(".");
					throw new AnalystError(result.toString());
				}
				break;
			case TypeDouble:
				Double.parseDouble(value);
				break;
			case typeFormat:
				if( ConvertStringConst.string2AnalystFileFormat(value)==null ) {
					StringBuilder result = new StringBuilder();
					result.append("Invalid file format for ");
					result.append(dotForm(section,subSection,name));
					result.append(", value is ");
					result.append(value);
					result.append(".");
					throw new AnalystError(result.toString());					
				}
				break;
			case TypeInteger:
				Integer.parseInt(value);
				break;
			case TypeListString:
				break;
			case TypeString:
				break;
			}
		} catch (NumberFormatException ex) {
			StringBuilder result = new StringBuilder();
			result.append("Illegal value for ");
			result.append(dotForm(section,subSection,name));
			result.append(", expecting a ");
			result.append(getEntryType().toString());
			result.append(", but got ");
			result.append(value);
			result.append(".");
			throw new AnalystError(result.toString());
		}
	}
}
