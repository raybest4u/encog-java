package org.encog.workbench;

import java.awt.*;

import javax.swing.*;

import org.encog.neural.persist.EncogPersistedCollection;
import org.encog.workbench.frames.EncogDocumentFrame;

public class EncogWorkBench {
	
	private static EncogWorkBench instance;
	private EncogDocumentFrame mainWindow;
	private EncogPersistedCollection currentFile;
	private String currentFileName;
	
	public static EncogWorkBench getInstance()
	{
		if( EncogWorkBench.instance == null )
		{
			EncogWorkBench.instance = new EncogWorkBench();
		}
		
		return EncogWorkBench.instance;
	}

	
	/**
	 * @return the mainWindow
	 */
	public EncogDocumentFrame getMainWindow() {
		return mainWindow;
	}



	/**
	 * @param mainWindow the mainWindow to set
	 */
	public void setMainWindow(EncogDocumentFrame mainWindow) {
		this.mainWindow = mainWindow;
	}
	
	



	/**
	 * @return the currentFile
	 */
	public EncogPersistedCollection getCurrentFile() {
		return currentFile;
	}


	/**
	 * @param currentFile the currentFile to set
	 */
	public void setCurrentFile(EncogPersistedCollection currentFile) {
		this.currentFile = currentFile;
	}


	/**
	 * @return the currentFileName
	 */
	public String getCurrentFileName() {
		return currentFileName;
	}


	/**
	 * @param currentFileName the currentFileName to set
	 */
	public void setCurrentFileName(String currentFileName) {
		this.currentFileName = currentFileName;
	}
	
	public static void load(String filename)
	{
		getInstance().setCurrentFileName(filename);
		getInstance().getCurrentFile().clear();
		getInstance().getCurrentFile().load(filename);
		getInstance().getMainWindow().redraw();
	}
	
	public static void save(String filename)
	{
		getInstance().setCurrentFileName(filename);
		getInstance().getCurrentFile().save(filename);
		getInstance().getMainWindow().redraw();		
	}
	
	public void close()
	{
		this.currentFile.clear();
		this.currentFileName = null;
		this.mainWindow.redraw();
	}
	
	public static void displayError(String title,String message)
	{
		JOptionPane.showMessageDialog(null, message,
				title, JOptionPane.ERROR_MESSAGE);
	}


	public static void main(String args[])
	{
		EncogWorkBench workBench = EncogWorkBench.getInstance();
		workBench.setMainWindow(new EncogDocumentFrame());
		
		if( args.length>0 )
		{
			EncogWorkBench.load(args[0]);
		}
		
		workBench.getMainWindow().setVisible(true);
	}


	public static void displayMessage(String title, String message) {
		JOptionPane.showMessageDialog(null, message,
				title, JOptionPane.INFORMATION_MESSAGE);
	}
	
	public static boolean askQuestion(String title,String question)
	{
		return JOptionPane.showConfirmDialog(getInstance().getMainWindow(), question,
				title, JOptionPane.YES_NO_OPTION)==JOptionPane.YES_OPTION;
	}
	
	public static Frame getCurrentFocus()
	{
	  Frame[] frames = JFrame.getFrames();
	  for(int i=0;i<frames.length;i++)
	  {
	    if(frames[i].hasFocus())
	    {
	      return frames[i];
	    }
	  }
	  return null;
	}
}
