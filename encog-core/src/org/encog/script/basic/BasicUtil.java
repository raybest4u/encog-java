package org.encog.script.basic;


public class BasicUtil {
	
	public static BasicKey findKeyword(String token)
	{
		for( BasicKey key : BasicKey.getKeys())
		{
			if( key.getName().equals(token))
				return key;
		}
		return null;
	}
	
	public static BasicKey  findKeyword(KeyNames token)
	{
		for( BasicKey key : BasicKey.getKeys())
		{
			if( key.getId()==token)
				return key;
		}
		return null;
	}
	public static void doInput(String str,String cap)
	{
		
	}
	
	public static int findKeyword(String str,String key)
	{
		boolean quote=false;
		int wptr;
		int rtn = 0;
		String cmp;

		cmp = key.toUpperCase();

		StringBuilder word = new StringBuilder();
		
		int index = 0;
		
		while(index<str.length())
		{
			char ch = str.charAt(index);
			
			if(ch=='\"')
			{
				quote=!quote;
				index++;
				continue;
			}

			if(quote)
			{
				index++;
				continue;
			}

			if( (ch=='\t') || (ch==' ') || (ch==13) || (ch==10) )
			{
				index++;
				if(word.toString().equals(cmp))
					return rtn;
				continue;
			}
			
			word.append(ch);
			rtn++;
			index++;
		}

		return -1;
	}


	public static void syncVariables(char opp,BasicVariable in1,BasicVariable in2,BasicVariable out1,BasicVariable out2)
	{
		out1.free();// Just in case
		out2.free();

		if( (in1.getObjectType()==BasicTypes.typeString) || (in2.getObjectType()==BasicTypes.typeString) )
		{
			out1.edit( in1.getStr());
			out2.edit( in2.getStr());
			return;
		}

		if( (in1.getObjectType()==BasicTypes.typeString) || (in2.getObjectType()==BasicTypes.typeString) )
		{
			out1.edit( in1.getStr());
			out2.edit( in2.getStr());
			return;
		}

		if( (in1.getObjectType()==BasicTypes.typeDouble) || (in2.getObjectType()==BasicTypes.typeDouble) )
		{
			out1.edit( in1.getDouble());
			out2.edit( in2.getDouble());
			return;
		}

		if( (in1.getObjectType()==BasicTypes.typeFloat) || (in2.getObjectType()==BasicTypes.typeFloat) )
		{
			out1.edit( in1.getFloat());
			out2.edit( in2.getFloat());
			return;
		}

		if( (in1.getObjectType()==BasicTypes.typeLong) || (in2.getObjectType()==BasicTypes.typeLong) )
		{
			out1.edit( in1.getLong());
			out2.edit( in2.getLong());
			return;
		}

		if( (in1.getObjectType()==BasicTypes.typeInteger) || (in2.getObjectType()==BasicTypes.typeInteger) )
		{
			out1.edit( in1.getShort());
			out2.edit( in2.getShort());
			return;
		}

		if( (in1.getObjectType()==BasicTypes.typeByte) || (in2.getObjectType()==BasicTypes.typeByte) )
		{
			out1.edit( in1.getByte());
			out2.edit( in2.getByte());
			return;
		}


		if( (in1.getObjectType()==BasicTypes.typeCharacter) || (in2.getObjectType()==BasicTypes.typeCharacter) )
		{
			out1.edit( in1.getShort());
			out2.edit( in2.getShort());
			return;
		}
				
		if( (in1.getObjectType()==BasicTypes.typeBoolean) || (in2.getObjectType()==BasicTypes.typeBoolean) )
		{
			out1.edit( in1.getBoolean());
			out2.edit( in2.getBoolean());
			return;
		}
	}

	public static void performAdd(BasicVariable target,BasicVariable var1,BasicVariable var2)
	{
		BasicVariable a = new BasicVariable();
		BasicVariable b = new BasicVariable();

		syncVariables('+',var1,var2,a,b);
		switch(a.getObjectType())
		{
		case typeString:
			StringBuilder temp = new StringBuilder();
			temp.append(a.getStr());
			temp.append(b.getStr());
			target.edit(temp.toString());
			break;
		case typeFloat:target.edit((float)a.getFloat()+b.getFloat());break;
		case typeInteger:target.edit((long)(a.getShort()+b.getShort()));break;
		case typeLong:target.edit((long)(a.getLong()+b.getLong()));break;
		case typeDouble:target.edit((double)a.getDouble()+b.getDouble());break;
		case typeByte:target.edit((byte)a.getByte()+b.getByte());break;
		case typeCharacter:target.edit((char)a.getShort()+b.getShort());break;
		default:
			throw(new BasicError(ErrorNumbers.errorType));
		}
	}

	public static void performSub(BasicVariable target,BasicVariable var1,BasicVariable var2)
	{
		BasicVariable a = new BasicVariable();
		BasicVariable b = new BasicVariable();

		syncVariables('-',var1,var2,a,b);
		switch(a.getObjectType())
		{
		case typeFloat:target.edit((float)a.getFloat()-b.getFloat());break;
		case typeInteger:target.edit((long)(a.getShort()-b.getShort()));break;
		case typeLong:target.edit((long)a.getLong()-b.getLong());break;
		case typeDouble:target.edit((double)a.getDouble()-b.getDouble());break;
		case typeByte:target.edit((byte)a.getByte()-b.getByte());break;
		case typeCharacter:target.edit((char)a.getShort()-b.getShort());break;
		default:
			throw(new BasicError(ErrorNumbers.errorType));
		}
	}

	public static void performMul(BasicVariable target,BasicVariable var1,BasicVariable var2)
	{
		BasicVariable a = new BasicVariable();
		BasicVariable b = new BasicVariable();

		syncVariables('*',var1,var2,a,b);
		switch(a.getObjectType())
		{
		case typeFloat:target.edit((float)a.getFloat()*b.getFloat());break;
		case typeInteger:target.edit((long)(a.getShort()*b.getShort()));break;
		case typeLong:target.edit((long)a.getLong()*b.getLong());break;
		case typeDouble:target.edit((double)a.getDouble()*b.getDouble());break;
		case typeByte:target.edit((byte)a.getByte()*b.getByte());break;
		case typeCharacter:target.edit((char)a.getShort()*b.getShort());break;
		default:
			throw(new BasicError(ErrorNumbers.errorType));
		}
	}

	public static void performDiv(BasicVariable target,BasicVariable var1,BasicVariable var2)
	{
		BasicVariable a = new BasicVariable();
		BasicVariable b = new BasicVariable();

		syncVariables('/',var1,var2,a,b);
		switch(a.getObjectType())
		{
		case typeFloat:target.edit((float)a.getFloat()/b.getFloat());break;
		case typeInteger:target.edit((long)(a.getShort()/b.getShort()));break;
		case typeLong:target.edit((long)a.getLong()/b.getLong());break;
		case typeDouble:target.edit((double)a.getDouble()/b.getDouble());break;
		case typeByte:target.edit((byte)a.getByte()/b.getByte());break;
		case typeCharacter:target.edit((char)a.getShort()/b.getShort());break;
		default:
			throw(new BasicError(ErrorNumbers.errorType));
		}
	}

	public static void performMod(BasicVariable target,BasicVariable var1,BasicVariable var2)
	{
		BasicVariable a = new BasicVariable();
		BasicVariable b = new BasicVariable();

		syncVariables('%',var1,var2,a,b);
		switch(a.getObjectType())
		{
		case typeFloat:
		case typeInteger:target.edit((long)a.getShort()%b.getShort());break;
		case typeDouble:
		case typeLong:target.edit((long)a.getLong()%b.getLong());break;
		case typeByte:target.edit((byte)a.getByte()%b.getByte());break;
		case typeCharacter:target.edit((char)a.getShort()%b.getShort());break;
		default:
			throw(new BasicError(ErrorNumbers.errorType));
		}
	}

	public static void performCat(BasicVariable target,BasicVariable var1,BasicVariable var2)
	{
		StringBuilder temp = new StringBuilder();
		temp.append(var1.getStr());
		temp.append(var2.getStr());
		target.edit(temp.toString());
		
	}

	public static void performPower(BasicVariable target,BasicVariable var1,BasicVariable var2)
	{
		if( (var1.getObjectType()==BasicTypes.typeString) ||
			(var1.getObjectType()==BasicTypes.typeBoolean) ||
			(var2.getObjectType()==BasicTypes.typeString) ||
			(var2.getObjectType()==BasicTypes.typeBoolean) )
		{
			throw(new BasicError(ErrorNumbers.errorType));
		}
		target.edit((double)Math.pow(var1.getDouble(),var2.getDouble()));

	}

	public static void performNeg(BasicVariable target,BasicVariable var)
	{
		BasicVariable a = new BasicVariable();
		BasicVariable b = new BasicVariable();

		switch(var.getObjectType())
		{
		case typeFloat:target.edit((float)-var.getFloat());break;
		case typeInteger:target.edit((long)-var.getShort());break;
		case typeLong:target.edit((long)-var.getLong());break;
		case typeDouble:target.edit((double)-var.getDouble());break;
		case typeByte:target.edit((byte)-var.getByte());break;
		case typeCharacter:target.edit((char)-var.getShort());break;
		default:
			throw(new BasicError(ErrorNumbers.errorType));
		}

	}

	public static String basicToUpper(String l) {
		StringBuilder result = new StringBuilder();
		boolean inQuote = false;
		
		for(int i=0;i<l.length();i++)
		{
			char ch = l.charAt(i);
			if( ch=='\"')
				inQuote = !inQuote;
			
			if( !inQuote )
				ch = Character.toUpperCase(ch);
			
			result.append(ch);
		}
		
		return result.toString();
	}
	
	public static int basicIndexOf(String str, int start, String srch)
	{
		boolean quote=false;
		int searchIndex = 0;
		
		for(int sourceIndex = start; sourceIndex<str.length()-srch.length(); sourceIndex++)
		{
			char ch = str.charAt(sourceIndex);
			char ch2 = srch.charAt(searchIndex);
			
			if( ch==34 )
				quote = !quote;
			
			if( !quote )
			{
				if( Character.toUpperCase(ch)==Character.toUpperCase(ch2))
				{
					searchIndex++;
					if( searchIndex==srch.length())
						return sourceIndex - srch.length();
				}
				else
					searchIndex = 0;
			}
		}
		
		return -1;
	}
}
