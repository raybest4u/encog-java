/*
 * Encog Workbench v2.x
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

package org.encog.workbench.frames.network;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.Polygon;

import org.encog.neural.networks.layers.Layer;
import org.encog.neural.networks.synapse.Synapse;
import org.encog.workbench.WorkbenchFonts;
import org.encog.workbench.frames.network.NetworkDiagram.Side;

public class DrawArrow {
	
	public static final int SELF_CONNECTED_WIDTH = 40;
	
		
	/**
	 * Generate a polygon for an arrow head that points the right way.
	 * @param side
	 * @param x
	 * @param y
	 * @return
	 */
	private static Polygon generateArrowHead(Side side, int x,int y)
	{
		int[] polyX = new int[3];
		int[] polyY = new int[3];
		
		switch(side)
		{
			case Top:
				polyX[0] = x-NetworkDiagram.ARROWHEAD_WIDTH/2;
				polyY[0] = y;
				polyX[1] = x+NetworkDiagram.ARROWHEAD_WIDTH/2;
				polyY[1] = y;
				polyX[2] = x;
				polyY[2] = y+NetworkDiagram.ARROWHEAD_WIDTH;
				break;
			case Bottom:
				polyX[0] = x-NetworkDiagram.ARROWHEAD_WIDTH/2;
				polyY[0] = y;
				polyX[1] = x+NetworkDiagram.ARROWHEAD_WIDTH/2;
				polyY[1] = y;
				polyX[2] = x;
				polyY[2] = y-NetworkDiagram.ARROWHEAD_WIDTH;
				break;
			case Right:
				polyX[0] = x;
				polyY[0] = y-NetworkDiagram.ARROWHEAD_WIDTH/2;
				polyX[1] = x;
				polyY[1] = y+NetworkDiagram.ARROWHEAD_WIDTH/2;
				polyX[2] = x-NetworkDiagram.ARROWHEAD_WIDTH;
				polyY[2] = y;
				break;
			case Left:
				polyX[0] = x;
				polyY[0] = y-NetworkDiagram.ARROWHEAD_WIDTH/2;
				polyX[1] = x;
				polyY[1] = y+NetworkDiagram.ARROWHEAD_WIDTH/2;
				polyX[2] = x+NetworkDiagram.ARROWHEAD_WIDTH;
				polyY[2] = y;
				break;
		}
		
		return new Polygon(polyX,polyY,3);
	}
	
	/**
	 * Draw an arrow, with an arrow head between the two points.
	 * @param g
	 * @param from
	 * @param to
	 * @param toSide
	 */
	private static void drawArrow(Graphics g,Point from,Point to, Side toSide)
	{
		Polygon arrowHead = generateArrowHead(toSide,to.x,to.y);
		
		g.drawLine(from.x,from.y,to.x,to.y);	
		g.fillPolygon(arrowHead);
	}
	
	public static void drawSelfArrow(Graphics g, Synapse synapse)
	{
		int x = synapse.getFromLayer().getX()+DrawLayer.LAYER_WIDTH-(SELF_CONNECTED_WIDTH/2);
		int y = synapse.getFromLayer().getY()-(SELF_CONNECTED_WIDTH/2);
		g.drawArc(x, y, SELF_CONNECTED_WIDTH, SELF_CONNECTED_WIDTH,300, 240);
		Polygon arrowHead = generateArrowHead(Side.Right,x+30,y+37);
		g.fillPolygon(arrowHead);
	}
	
	
	
	public static void drawArrow(Graphics g,Synapse synapse, String label)
	{
		CalculateArrow arrow = new CalculateArrow(synapse, true);
		
		drawArrow(g,arrow.getFrom(),arrow.getTo(),arrow.getToSide());
		drawLabel(g,arrow.getFrom(),arrow.getTo(),label);
	}
	
	private static void drawLabel(Graphics g, Point from, Point to, String label)
	{
		int labelX = Math.min((int)from.getX(), (int)to.getX());
		int labelY = Math.min((int)from.getY(), (int)to.getY());
		int distX = Math.abs((int)from.getX() - (int)to.getX())/2;
		int distY = Math.abs((int)from.getY() - (int)to.getY())/2;
		
		g.setFont(WorkbenchFonts.getTextFont());
		g.drawString(label, labelX+distX, labelY+distY);
	}
}