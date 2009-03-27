package org.encog.util.xml;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Queue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.xml.transform.Result;
import javax.xml.transform.Transformer;
import javax.xml.transform.sax.TransformerHandler;

import org.encog.EncogError;
import org.encog.util.xml.XMLElement.XMLElementType;
import org.xml.sax.Attributes;
import org.xml.sax.InputSource;
import org.xml.sax.Locator;
import org.xml.sax.SAXException;

import com.sun.org.apache.xerces.internal.parsers.SAXParser;


public class XMLRead implements Runnable{
	
	
	class XMLReadHandler implements TransformerHandler
	{
		private String systemID;
		private Transformer transformer;
		private Locator locator;


		public void setResult(Result arg0) throws IllegalArgumentException {
			// TODO Auto-generated method stub
			
		}		

		public void characters(char[] ch, int start, int length) 
				throws SAXException {
			try {
				String text = new String(ch,start,length);
				XMLElement element = new XMLElement(XMLElementType.text,text);
				queue.put(element);
			} catch (InterruptedException e) {
			}
		}


		public void endDocument() throws SAXException {
			
			try {
				done.set(true);
				queue.put(new XMLElement(XMLElementType.documentEnd));
			} catch (InterruptedException e) {
			}
		}

		public void endElement(String uri, String localName, String qName) 
				throws SAXException {
			
			try {
				XMLElement element = new XMLElement(XMLElementType.end,localName);
				queue.put(element);
			} catch (InterruptedException e) {

			}
		}

		public void endPrefixMapping(String prefix) throws SAXException {
			// TODO Auto-generated method stub
			
		}

		public void ignorableWhitespace(char[] ch, int start, int length) 
				throws SAXException {
			// TODO Auto-generated method stub
			
		}

		public void processingInstruction(String target, String data)
				throws SAXException {
			// TODO Auto-generated method stub
			
		}

		public void setDocumentLocator(Locator locator) {
			this.locator = locator;
			
		}

		public void skippedEntity(String name) throws SAXException {
			// TODO Auto-generated method stub
			
		}

		public void startDocument() throws SAXException {
			try {
				queue.put(new XMLElement(XMLElementType.documentBegin));
			} catch (InterruptedException e) {

			}
			
		}

		public void startElement(String uri, String localName, String qName, Attributes atts)  throws SAXException {
			XMLElement element = new XMLElement(XMLElementType.start,localName);
			for(int i=0;i<atts.getLength();i++)
			{
				String name = atts.getLocalName(i);
				String value = atts.getValue(i);
				element.getAttributes().put(name, value);
			}
			
			try {
				queue.put(element);
			} catch (InterruptedException e) {
			}
		}

		public void startPrefixMapping(String prefix, String uri)
				throws SAXException {
			// TODO Auto-generated method stub
			
		}

		public void comment(char[] ch, int start, int length) 
				throws SAXException {
			// TODO Auto-generated method stub
			
		}

		public void endCDATA() throws SAXException {

			
		}

		public void endDTD() throws SAXException {
			// TODO Auto-generated method stub
			
		}

		public void endEntity(String entity) throws SAXException {

			
		}

		public void startCDATA() throws SAXException {

			
		}

		public void startDTD(String arg0, String arg1, String arg2)
				throws SAXException {
			// TODO Auto-generated method stub
			
		}

		public void startEntity(String entity) throws SAXException {

			
		}

		public void notationDecl(String name, String publicId, String systemId) 
				throws SAXException {
			// TODO Auto-generated method stub
			
		}

		public void unparsedEntityDecl(String name, String publicId, String systemId, String notationName) 
			throws SAXException {
			// TODO Auto-generated method stub			
		}

		/**
		 * @return the systemID
		 */
		public String getSystemId() {
			return systemID;
		}

		/**
		 * @return the transformer
		 */
		public Transformer getTransformer() {
			return transformer;
		}

		/**
		 * @param transformer the transformer to set
		 */
		public void setTransformer(Transformer transformer) {
			this.transformer = transformer;
		}


		public void setSystemId(String arg0) {
			// TODO Auto-generated method stub
			
		}
		
		
		
	}
	
	private InputStream input;
	private XMLReadHandler handler;
	private InputSource inputSource;
	private SAXParser parser;
	private Thread thread;
	private BlockingQueue<XMLElement> queue;
	private AtomicBoolean done = new AtomicBoolean();
	
	public XMLRead(InputStream input)
	{
			this.input = input;
			this.done.set(false);
			this.queue = new ArrayBlockingQueue<XMLElement>(10);
			this.handler = new XMLReadHandler();
			
			this.inputSource = new InputSource(this.input);
			this.parser = new SAXParser();
			parser.setContentHandler(this.handler);
			this.thread = new Thread(this);
			this.thread.start();

	}
	
	public void close()
	{
		try {
			this.input.close();
		} catch (IOException e) {
			throw(new EncogError(e));
		}
	}
	
	public XMLElement get()
	{
		try {
			if( this.queue.size()<1 && this.done.get() )
				return null;
			else
				return this.queue.take();
		} catch (InterruptedException e) {
			return null;
		}
	}

	public void run() {
		try {
			parser.parse(this.inputSource);
		} catch (SAXException e) {
			throw new EncogError(e);
		} catch (IOException e) {
			throw new EncogError(e);
		}
		
	}

	public void skipObject(XMLElement element) {
		XMLElement next;
		
		while( (next = get())!=null )
		{
			XMLElementType type = next.getType();
			if( type== XMLElementType.start )
			{
				skipObject(next);
			}
			else if( type==XMLElementType.end )
				return;
		}
		
	}

}
