package org.encog.parse.recognize;

import java.util.ArrayList;
import java.util.List;

import org.encog.parse.signal.Signal;


public class Recognize {
  private List<RecognizeElement> pattern = new ArrayList<RecognizeElement>();
  private int index = 0;
  private String type;
  private int startIndex = -1;
  private int stopIndex;
  private int currentIndex = 0;
  private boolean ignore = false;
  private Class signalClass = Signal.class;

  public Recognize(String type)
  {
    this.type = type;
  }

  public void add(RecognizeElement re)
  {
    pattern.add(re);
  }

  public void setIgnore(boolean ignore)
  {
    this.ignore = ignore;
  }

  public boolean getIgnore()
  {
    return this.ignore;
  }

  private void startTracking()
  {
//System.out.println("Start tracking");
    if (startIndex==-1)
      startIndex = currentIndex;
    stopIndex = currentIndex+1;
  }

  private void stopTracking()
  {
//System.out.println("Stop tracking");
    startIndex=-1;
    index=0;
  }

  public boolean recognize(Signal signal)
  {
    try {
      boolean found;
      do {
        found = recognizeIteration(signal);
      } while (found);
      return found;
    } catch (InstantiationException e) {
      System.out.println(e);
    } catch (IllegalAccessException e) {
      System.out.println(e);
    }
    return false;
  }  

  protected boolean recognizeIteration(Signal signal)
  throws InstantiationException,IllegalAccessException
  {
    startIndex = -1;
    index = 0;
    currentIndex = 0;

    Object array[] = signal.getData().toArray();
    while ( currentIndex<array.length ) {

      RecognizeElement re = pattern.get(index);
      Signal signalElement = (Signal)array[currentIndex];

      if (signalElement.getIgnore()) {
        currentIndex++;
        continue;
      }

//System.out.println("Recognize Element:" + signalElement.dump() );
      boolean success = re.recognize(signalElement);



      switch (re.getAllow()) {
      case RecognizeElement.ALLOW_ONE:
        if (success) {
          startTracking();
          index++;
        } else {
          stopTracking();
        }
        break;

      case RecognizeElement.ALLOW_MULTIPLE:
        if (success) {
          startTracking();
        } else
          index++;

        break;
      }


      if (index>=pattern.size()) {


        if (startIndex!=-1) {
          Signal temp = signal.pack(startIndex,stopIndex,type,getSignalClass());
          temp.setName(re.getName());
          temp.setIgnore(ignore);
          return true;
        }
        index=0;
      }
      currentIndex++;

    }
    return false;
  }


  public RecognizeElement createElement(int allow)
  {
    RecognizeElement result = new RecognizeElement(allow);
    add(result);
    return result;
  }

  public Class getSignalClass()
  {
    return signalClass;
  }

  public void setSignalClass(Class signalClass)
  {
    this.signalClass = signalClass;
  }
}
