package org.encog.workbench.frames.network;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Polygon;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;

import org.encog.EncogError;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.layers.ContextLayer;
import org.encog.neural.networks.layers.Layer;
import org.encog.neural.networks.layers.RadialBasisFunctionLayer;
import org.encog.neural.networks.synapse.DirectSynapse;
import org.encog.neural.networks.synapse.OneToOneSynapse;
import org.encog.neural.networks.synapse.Synapse;
import org.encog.neural.networks.synapse.SynapseType;
import org.encog.neural.networks.synapse.WeightedSynapse;
import org.encog.neural.networks.synapse.WeightlessSynapse;
import org.encog.neural.prune.PruneSelective;
import org.encog.util.math.rbf.GaussianFunction;
import org.encog.util.math.rbf.RadialBasisFunction;
import org.encog.workbench.EncogWorkBench;
import org.encog.workbench.WorkBenchError;
import org.encog.workbench.WorkbenchFonts;
import org.encog.workbench.dialogs.layers.EditBasicLayer;
import org.encog.workbench.dialogs.layers.EditContextLayer;
import org.encog.workbench.dialogs.layers.EditRadialLayer;
import org.encog.workbench.frames.MatrixFrame;
import org.encog.workbench.frames.network.NetworkTool.Type;
import org.encog.workbench.util.MouseUtil;

public class NetworkDiagram extends JPanel implements MouseListener, MouseMotionListener, ActionListener {

	enum Side
	{
		Top,
		Bottom,
		Left,
		Right
	}
	
	
	public static final int VIRTUAL_WIDTH = 2000;
	public static final int VIRTUAL_HEIGHT = 2000;
	public static final int ARROWHEAD_WIDTH = 10;
	private final NetworkFrame parent;
	private Layer selected;
	private Layer fromLayer;
	private int dragOffsetX;
	private int dragOffsetY;
	private Image offscreen;
	private Graphics offscreenGraphics;
	private List<Layer> layers = new ArrayList<Layer>();
	private Set<Layer> orphanLayers = new HashSet<Layer>();
	private JPopupMenu popupNetworkLayer;
	private JMenuItem popupNetworkLayerDelete;
	private JMenuItem popupNetworkLayerEdit;
	private JPopupMenu popupNetworkSynapse;
	private JMenuItem popupNetworkSynapseDelete;
	private JMenuItem popupNetworkSynapseMatrix;
	private Synapse selectedSynapse;
	
	public NetworkDiagram(NetworkFrame parent)
	{
		this.parent = parent;
		this.addMouseListener(this);
		this.addMouseMotionListener(this);
		this.setPreferredSize(new Dimension(VIRTUAL_HEIGHT,VIRTUAL_WIDTH));
		getLayers();
		
		this.popupNetworkLayer = new JPopupMenu();
		this.popupNetworkLayerEdit = addItem(this.popupNetworkLayer,
				"Edit Layer", 'e');
		this.popupNetworkLayerDelete = addItem(this.popupNetworkLayer,
				"Delete Layer", 'd');
		
		this.popupNetworkSynapse = new JPopupMenu();
		this.popupNetworkSynapseDelete = addItem(this.popupNetworkSynapse,
				"Delete Synapse", 'd');
		this.popupNetworkSynapseMatrix = addItem(this.popupNetworkSynapse,
				"Edit Matrix", 'm');
	}
	
	private void obtainOffScreen()
	{
		if( this.offscreen==null)
		{
			this.offscreen = this.createImage(VIRTUAL_WIDTH, VIRTUAL_HEIGHT);
			this.offscreenGraphics = this.offscreen.getGraphics();
		}
	}
	
	public void paint(Graphics g)
	{
		obtainOffScreen();
		offscreenGraphics.setColor(Color.WHITE);
		offscreenGraphics.fillRect(0,0,getWidth(),getHeight());
		offscreenGraphics.setColor(Color.BLACK);
		
		BasicNetwork network = (BasicNetwork)this.parent.getEncogObject();
		for(Layer layer: this.layers)
		{
			// draw any synapse arrows
			for(Synapse synapse: layer.getNext() )
			{
				if( synapse.isSelfConnected() )
					drawSelfConnectedSynapse(offscreenGraphics,synapse);
				else
					drawSynapse(offscreenGraphics,synapse);
			}
			
			// draw the actual layer
			DrawLayer.drawLayer(this, offscreenGraphics,layer);
			
			if(network.isInput(layer))
			{
				drawInput(offscreenGraphics,layer);
			}
			if(network.isOutput(layer))
			{
				drawOutput(offscreenGraphics,layer);
			}
			
			if( this.selected==layer)
			{
				DrawLayer.drawSelection(offscreenGraphics,layer);
			}
			if( this.fromLayer==layer)
			{
				DrawLayer.drawFromSelection(offscreenGraphics,layer);
			}
		}
		
		g.drawImage(this.offscreen, 0,0,this);
		
	}
	
	private void drawSelfConnectedSynapse(Graphics g,
			Synapse synapse) {
		
		if( synapse==this.selectedSynapse)
			g.setColor(Color.CYAN);
		else
			g.setColor(Color.BLACK);
		
		DrawArrow.drawSelfArrow(g, synapse);
		g.drawString(this.getSynapseText(synapse), 
				synapse.getToLayer().getX()+100, 
				synapse.getFromLayer().getY()-20);
		
	}
	
	private String getSynapseText(Synapse synapse)
	{
		if( synapse instanceof WeightedSynapse )
		{
			return "Weighted";
		}
		else if( synapse instanceof WeightlessSynapse )
		{
			return "Weightless";
		}
		else if( synapse instanceof OneToOneSynapse )
		{
			return "1:1";
		}
		else if( synapse instanceof DirectSynapse )
		{
			return "Direct";
		}
		else
			return "Unknown";
	}

	private void drawSynapse(Graphics g, Synapse synapse) {
		
		if( synapse==this.selectedSynapse )
			g.setColor(Color.CYAN);
		else
			g.setColor(Color.BLACK);
		
		String type = getSynapseText(synapse);
		DrawArrow.drawArrow(g,synapse,type);		
	}

	
	

	
	private Layer findLayer(MouseEvent e)
	{
		// was a layer something clicked
		Layer clickedLayer = null;
		for(int i=layers.size()-1;i>=0;i--)
		{
			Layer layer = layers.get(i);
			if( contains(layer,e.getX(),e.getY()))
			{
				clickedLayer = layer;
			}
		}
		return clickedLayer;
	}
	
	private Synapse findSynapse(MouseEvent e)
	{
		for(Layer layer: this.layers)
		{
			for(Synapse synapse: layer.getNext())
			{
				CalculateArrow arrow = new CalculateArrow(synapse,false);
				Polygon p = arrow.obtainPologygon();
				if( p.contains(e.getX(),e.getY()))
					return synapse;
			}
		}
		return null;
	}

	public void mouseClicked(MouseEvent e) {
		
		Layer clickedLayer = findLayer(e);
		
		if (MouseUtil.isRightClick(e)) {
			rightClick(e);
		}
		else
		if (e.getClickCount() == 2) {

			doubleClick(e,clickedLayer);
		}
		
	}
	
	private void rightClick(MouseEvent e)
	{
		int x = e.getX(); 
		int y = e.getY();
		
		Layer clickedLayer = this.findLayer(e);
		
		if( clickedLayer!=null )
		{	
			this.popupNetworkLayer.show(e.getComponent(), x, y);
		}
		else
		{
			Synapse synapse = this.findSynapse(e);
			if( synapse!=null )
			{
				this.popupNetworkSynapse.show(e.getComponent(), x, y);
			}
		}
	}
	
	private void doubleClick(MouseEvent e,Layer clickedLayer)
	{
		
	}

	public void mouseEntered(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}

	public void mouseExited(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}

	@SuppressWarnings("unchecked")
	public void mousePressed(MouseEvent e) {
		
		Layer clickedLayer = findLayer(e);
		
		BasicNetwork network = (BasicNetwork)this.parent.getEncogObject();
		
		// was no layer clicked?  Maybe a synapse was.
		if( clickedLayer==null )
		{
			Synapse clickedSynapse = findSynapse(e);
			if( clickedSynapse!=null)
			{
				clearSelection();
				this.selectedSynapse = clickedSynapse;
				repaint();
				return;
			}
		}
		
		// is a synapse connection about to start or end
		if( this.parent.getNetworkToolbar().getSelected()!=null)
		{
			// is it a synapse
			if( this.parent.getNetworkToolbar().getSelected().getType()==Type.synapse)
			{
				// about to start
				if( this.fromLayer==null )
				{
					clearSelection();
					this.selected = this.fromLayer = clickedLayer;
					repaint();
					return;
				}
				else
				{
				// about to create
					createSynapse(clickedLayer);
					return;
				}
			}
		}
		
		
		// was something deselected or selected
		
		if(clickedLayer!=null)
		{
			if( selected==clickedLayer && !MouseUtil.isRightClick(e))
			{
				// deselected
				this.clearSelection();
				return;
			}
			else
			{
				// selected
				clearSelection();
				selected = clickedLayer;
				dragOffsetX = e.getX()-clickedLayer.getX();
				dragOffsetY = e.getY()-clickedLayer.getY();
				repaint();
				return;
			}
		}
		
		
		// nothing was selected, is there a toolbar item that needs to be added
		if( this.parent.getNetworkToolbar().getSelected()!=null && 
				this.parent.getNetworkToolbar().getSelected().getType()==Type.synapse )
		{
			EncogWorkBench.displayError("Error", "Can't drop a synapse there, chose a 'from layer'\nthen a 'to layer'.");
		}
		else if( this.parent.getNetworkToolbar().getSelected()!=null)
		{
			try {
				clearSelection();
				Class<? extends Layer> c = this.parent.getNetworkToolbar().getSelected().getClassType();
				Layer layer = (Layer)c.newInstance();
				this.parent.getNetworkToolbar().setSelected(null);
				
				if( network.getInputLayer()==null )
				{
					network.addLayer(layer);
					network.getStructure().finalizeStructure();
				}
				else
					this.orphanLayers.add(layer);
				layer.setX(e.getX());
				layer.setY(e.getY());
				this.getLayers();
				this.selected = layer;
				this.parent.getNetworkToolbar().setSelected(null);
				repaint();
			} catch (InstantiationException e1) {
				throw new WorkBenchError(e1);
			} catch (IllegalAccessException e1) {
				throw new WorkBenchError(e1);
			}
			
		}
		
		// nothing was selected, deselect if something was previously
		if( this.selected!=null || this.selectedSynapse!=null)
		{
			this.clearSelection();
		}
		
	}
	
	private void createSynapse(Layer clickedLayer)
	{
		BasicNetwork network = (BasicNetwork)this.parent.getEncogObject();
		
		// validate any obvious errors
		if( this.fromLayer.isConnectedTo(clickedLayer))
		{
			EncogWorkBench.displayError("Can't Create Synapse", "There is already a synapse between these two layers.\nYou must delete it first.");
			this.parent.clearSelection();
			return;
		}
		
		// try to create it
		try
		{
		NetworkTool tool = this.parent.getNetworkToolbar().getSelected();
		if( tool.getClassType() == WeightedSynapse.class )
		{
			this.fromLayer.addNext(clickedLayer,SynapseType.Weighted);
		}
		else if( tool.getClassType() == WeightlessSynapse.class )
		{
			this.fromLayer.addNext(clickedLayer,SynapseType.Weightless);
		}
		else if( tool.getClassType() == DirectSynapse.class )
		{
			this.fromLayer.addNext(clickedLayer,SynapseType.Direct);
		}
		else if( tool.getClassType() == OneToOneSynapse.class )
		{
			this.fromLayer.addNext(clickedLayer,SynapseType.OneToOne);
		}
		}
		catch(EncogError e)
		{
			EncogWorkBench.displayError("Synapse Error", e.getMessage());
		}
		
		// Attempt to determine the output layer
		network.inferOutputLayer();
		
		// recreate the network
		
		network.getStructure().finalizeStructure();
		this.parent.clearSelection();
		repaint();
	}
	
	private boolean contains(Layer layer,int x, int y)
	{
		return( x>layer.getX() && (x<layer.getX()+DrawLayer.LAYER_WIDTH) &&
			y>layer.getY() && (y<layer.getY()+DrawLayer.LAYER_HEIGHT) );
	}

	public void mouseReleased(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	
	private void getLayers()
	{
		BasicNetwork network = (BasicNetwork)this.parent.getEncogObject();
		
		network.getStructure().finalizeStructure();
		
		// first remove any orphans that may have made it into the "real" list
		fixOrphans();
		
		// now build the layer list
		this.layers.clear();
		this.layers.addAll(network.getStructure().getLayers());
		this.layers.addAll(this.orphanLayers);
	}

	public void mouseDragged(MouseEvent e) {
		
		if(this.selected!=null)
		{
			this.selected.setX(e.getX()-dragOffsetX);
			this.selected.setY(e.getY()-dragOffsetY);
			repaint();
		}
		
	}

	public void mouseMoved(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	public void clearSelection() {
		if(this.selected!=null || this.selectedSynapse!=null)
		{
			this.fromLayer = null;
			this.selected=null;
			this.selectedSynapse=null;
			repaint();
		}
		
	}
	

	public void actionPerformed(final ActionEvent action) {
		
		if (action.getSource() == this.popupNetworkLayerEdit) {
			if(this.selected instanceof ContextLayer)
				performContextLayerEdit();
			else if(this.selected instanceof RadialBasisFunctionLayer)
			performRadialLayerEdit();
			else if(this.selected instanceof BasicLayer)
				performBasicLayerEdit();
			
		} else if (action.getSource() == this.popupNetworkLayerDelete) {
			performLayerDelete();
		}  
		else if (action.getSource() == this.popupNetworkSynapseDelete) {
			performSynapseDelete();
		}  else if( action.getSource() == this.popupNetworkSynapseMatrix ) {
			performSynapseMatrix();
		}
	}
		
	private void performSynapseMatrix() {
		MatrixFrame frame = new MatrixFrame((BasicNetwork)this.parent.getEncogObject(),this.selectedSynapse);
		frame.setVisible(true);
		
	}

	public JMenuItem addItem(final JPopupMenu m, final String s,
			final int key) {

		final JMenuItem mi = new JMenuItem(s, key);
		mi.addActionListener(this);
		m.add(mi);
		return mi;
	}
	
	private void drawInput(Graphics g, Layer layer)
	{
		String str = ">>>Input>>>";
		
		g.setFont(WorkbenchFonts.getTextFont());
		FontMetrics fm = g.getFontMetrics();
		
		int height = fm.getHeight();
		int width = fm.stringWidth(str);
		int x = layer.getX();
		int y = layer.getY()-height;
		
		int center = (DrawLayer.LAYER_WIDTH/2)-(width/2);
		g.drawString(str, x+center,y+height-3);
		g.drawRect(x, y, DrawLayer.LAYER_WIDTH, height);
	}
	
	private void drawOutput(Graphics g, Layer layer)
	{
		String str = "<<<Output<<<";
		
		g.setFont(WorkbenchFonts.getTextFont());
		FontMetrics fm = g.getFontMetrics();
		
		int height = fm.getHeight();
		int width = fm.stringWidth(str);
		int x = layer.getX();
		int y = layer.getY();
		
		int center = (DrawLayer.LAYER_WIDTH/2)-(width/2);
		g.drawString(str, x+center,y+height-3+DrawLayer.LAYER_HEIGHT);
		g.drawRect(x, y+DrawLayer.LAYER_HEIGHT, DrawLayer.LAYER_WIDTH, height);
	}
	
	public void performSynapseDelete()
	{
		BasicNetwork network = (BasicNetwork)this.parent.getEncogObject();
		
		if( this.selectedSynapse!=null && 
				EncogWorkBench.askQuestion("Are you sure?", "Do you want to delete this synapse?") )
		{
			// add all layers to orphan layers, some will be removed later
			this.orphanLayers.addAll(network.getStructure().getLayers());
			
			// perform the delete
			this.selectedSynapse.getFromLayer().getNext().remove(this.selectedSynapse);
			network.getStructure().finalizeStructure();
			
			// handle any orphans
			this.fixOrphans();
			network.inferOutputLayer();
			
			// rebuild the network
			getLayers();
			
			// does the output layer need to be adjusted
			this.repaint();
		}
	}
	
	public void performLayerDelete()
	{
		if( EncogWorkBench.askQuestion("Are you sure?", "Do you want to delete this layer?") )
		{
			BasicNetwork network = (BasicNetwork)this.parent.getEncogObject();
			
			// add all layers to orphan layers, some will be removed later
			this.orphanLayers.addAll(network.getStructure().getLayers());
			
			// are we removing the input layer?
			if( this.selected==network.getInputLayer())
			{
				if( this.selected!=null && this.selected.getNext().size()>0 )
				{
				Synapse nextSynapse = this.selected.getNext().get(0);
				Layer nextLayer = nextSynapse.getToLayer();
				network.setInputLayer(nextLayer);
				}
				else
				{
					network.setInputLayer(null);
					network.setOutputLayer(null);
				}
			}
			
			// remove any synapses to this layer
			for(Synapse synapse: network.getStructure().getSynapses() )
			{
				if( synapse.getToLayer()==this.selected)
				{
					synapse.getFromLayer().getNext().remove(synapse);
				}
			}
			
			// rebuild the network & attempt to determine the output layer
			network.inferOutputLayer();
			network.getStructure().finalizeStructure();
			
			// fix the orphan list
			fixOrphans();
			this.orphanLayers.remove(this.selected);
			this.getLayers();
			
			// redraw
			this.clearSelection();
			repaint();
		}
	}

	private void fixOrphans() {
		
		BasicNetwork network = (BasicNetwork)this.parent.getEncogObject();
		
		for(Layer layer: network.getStructure().getLayers() )
		{
			this.orphanLayers.remove(layer);
		}
		
	}
	
	private void performBasicLayerEdit() {
		EditBasicLayer dialog = new EditBasicLayer(this.parent, this.selected);
		dialog.setActivationFunction(this.selected.getActivationFunction());
		dialog.getNeuronCount().setValue(this.selected.getNeuronCount());
		dialog.getUseThreshold().setValue(this.selected.hasThreshold());
		dialog.setTableEditable();
		
		for(int i=0;i<this.selected.getNeuronCount();i++)
		{
			dialog.getThresholdTable().setValue(i,0,"#"+(i+1));
			if(this.selected.hasThreshold())
				dialog.getThresholdTable().setValue(i,1,""+this.selected.getThreshold(i));
			else
				dialog.getThresholdTable().setValue(i,1,"N/A");
		}
				
		if( dialog.process() )
		{
			// were thresholds added or removed
			if( !dialog.getUseThreshold().getValue())
			{
				// removed
				this.selected.setThreshold(null);
			}
			else
			{
				// add thresholds if needed
				if( dialog.getUseThreshold().getValue() && !this.selected.hasThreshold())
				{
					this.selected.setThreshold(new double[this.selected.getNeuronCount()]);
				}
			}
			
			// did the neuron count change?
			if( dialog.getNeuronCount().getValue()!=this.selected.getNeuronCount())
			{
				PruneSelective prune = new PruneSelective((BasicNetwork)this.parent.getEncogObject());
				prune.changeNeuronCount(this.selected, dialog.getNeuronCount().getValue());
			}
			else
			{
				// if the neuron count did not change, copy any new threshold values
				boolean updateThreshold = dialog.getUseThreshold().getValue();
				
				if(updateThreshold)
				{
				for(int i=0;i<this.selected.getNeuronCount();i++)
				{
					double value = 0;
					try
					{
						value = Double.parseDouble(dialog.getThresholdTable().getValue(i,1));
					}
					catch(NumberFormatException e)
					{
						// just let the value go to zero
					}
					this.selected.setThreshold(i, value);
				}
				}
			}
			
			this.selected.setActivationFunction(dialog.getActivationFunction());
			repaint();
			
		}
		
	}
	
	private void performContextLayerEdit() {
		ContextLayer contextLayer = (ContextLayer)this.selected;
		EditContextLayer dialog = new EditContextLayer(this.parent, contextLayer);
		dialog.setActivationFunction(this.selected.getActivationFunction());
		dialog.getNeuronCount().setValue(this.selected.getNeuronCount());
		
		
			for(int i=0;i<this.selected.getNeuronCount();i++)
			{
				dialog.getThresholdTable().setValue(i,0,"#"+(i+1));
				if( contextLayer.hasThreshold())
					dialog.getThresholdTable().setValue(i,1,""+contextLayer.getThreshold(i));
				else
					dialog.getThresholdTable().setValue(i,1,"N/A");
				dialog.getThresholdTable().setValue(i,2,""+contextLayer.getContext().getData(i));
			}
		
		if(this.selected.hasThreshold())
		{
			dialog.getUseThreshold().setValue(true);
		}
		else
		{
			dialog.getUseThreshold().setValue(false);
		}
		
		dialog.setTableEditable();
		if( dialog.process() )
		{
			// were thresholds added or removed
			if( !dialog.getUseThreshold().getValue())
			{
				// removed
				this.selected.setThreshold(null);
			}
			else
			{
				// add thresholds if needed
				if( dialog.getUseThreshold().getValue() && !this.selected.hasThreshold())
				{
					this.selected.setThreshold(new double[this.selected.getNeuronCount()]);
				}
			}
			
			// did the neuron count change?
			if( dialog.getNeuronCount().getValue()!=this.selected.getNeuronCount())
			{
				PruneSelective prune = new PruneSelective((BasicNetwork)this.parent.getEncogObject());
				prune.changeNeuronCount(this.selected, dialog.getNeuronCount().getValue());
			}
			else
			{
				// if the neuron count did not change, copy threshold and context values
				boolean updateThreshold = dialog.getUseThreshold().getValue();
				for(int i=0;i<this.selected.getNeuronCount();i++)
				{
					double thresholdValue = 0;
					double contextValue = 0;
					try
					{
						if( updateThreshold )
						thresholdValue = Double.parseDouble(dialog.getThresholdTable().getValue(i,1));
						contextValue = Double.parseDouble(dialog.getThresholdTable().getValue(i,2));
					}
					catch(NumberFormatException e)
					{
						// just let the value go to zero
					}
					if( updateThreshold )
					this.selected.setThreshold(i, thresholdValue);
					contextLayer.getContext().setData(i, contextValue);
				}
			}
			
			this.selected.setActivationFunction(dialog.getActivationFunction());
			repaint();
			
		}
		
	}
	
	private void performRadialLayerEdit() {
		RadialBasisFunctionLayer radialLayer = (RadialBasisFunctionLayer)this.selected;
		EditRadialLayer dialog = new EditRadialLayer(this.parent, radialLayer);
		dialog.getNeuronCount().setValue(this.selected.getNeuronCount());
		for(int i=0;i<radialLayer.getRadialBasisFunction().length;i++)
		{
			RadialBasisFunction rbf = radialLayer.getRadialBasisFunction()[i];
			dialog.getRadial().getModel().setValueAt(""+(i+1), i, 0);
			dialog.getRadial().getModel().setValueAt(""+rbf.getCenter(), i, 1);
			dialog.getRadial().getModel().setValueAt(""+rbf.getPeak(), i, 2);
			dialog.getRadial().getModel().setValueAt(""+rbf.getWidth(), i, 3);
		}
				
		dialog.getChart().refresh();
		if( dialog.process() )
		{			
			
			if( dialog.getNeuronCount().getValue()!=this.selected.getNeuronCount())
			{
				// did the neuron count change?
				PruneSelective prune = new PruneSelective((BasicNetwork)this.parent.getEncogObject());
				prune.changeNeuronCount(this.selected, dialog.getNeuronCount().getValue());
			}
			else
			{
				// update the RBF's
				for(int i=0;i<radialLayer.getRadialBasisFunction().length;i++)
				{
					double center = Double.parseDouble(dialog.getRadial().getValue(i, 1));
					double peak = Double.parseDouble(dialog.getRadial().getValue(i, 2));
					double width = Double.parseDouble(dialog.getRadial().getValue(i, 3));
					radialLayer.getRadialBasisFunction()[i] = new GaussianFunction(center,peak,width);
					
				}
			}
			repaint();
		}
		
	}
	
	public NetworkFrame getNetworkFrame()
	{
		return this.parent;
	}
	
}
