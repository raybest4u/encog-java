package org.encog.neural.bam;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;

import org.encog.persist.EncogFileSection;
import org.encog.persist.EncogPersistor;
import org.encog.persist.EncogReadHelper;
import org.encog.persist.EncogWriteHelper;
import org.encog.persist.PersistConst;

/**
 * Persist the BAM network.
 */
public class PersistBAM implements EncogPersistor {

	/**
	 * {@inheritDoc}
	 */
	@Override
	public final int getFileVersion() {
		return 1;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public final String getPersistClassString() {
		return "BAM";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public final Object read(final InputStream is) {
		final BAM result = new BAM();
		final EncogReadHelper in = new EncogReadHelper(is);
		EncogFileSection section;

		while ((section = in.readNextSection()) != null) {
			if (section.getSectionName().equals("BAM")
					&& section.getSubSectionName().equals("PARAMS")) {
				final Map<String, String> params = section.parseParams();
				result.getProperties().putAll(params);
			}
			if (section.getSectionName().equals("BAM")
					&& section.getSubSectionName().equals("NETWORK")) {
				final Map<String, String> params = section.parseParams();

				result.setF1Count(EncogFileSection.parseInt(params,
						PersistConst.PROPERTY_F1_COUNT));
				result.setF2Count(EncogFileSection.parseInt(params,
						PersistConst.PROPERTY_F2_COUNT));
				result.setWeightsF1toF2(EncogFileSection.parseMatrix(params,
						PersistConst.PROPERTY_WEIGHTS_F1_F2));
				result.setWeightsF2toF1(EncogFileSection.parseMatrix(params,
						PersistConst.PROPERTY_WEIGHTS_F2_F1));
			}
		}

		return result;

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public final void save(final OutputStream os, final Object obj) {
		final EncogWriteHelper out = new EncogWriteHelper(os);
		final BAM bam = (BAM) obj;
		out.addSection("BAM");
		out.addSubSection("PARAMS");
		out.addProperties(bam.getProperties());
		out.addSubSection("NETWORK");

		out.writeProperty(PersistConst.PROPERTY_F1_COUNT, bam.getF1Count());
		out.writeProperty(PersistConst.PROPERTY_F2_COUNT, bam.getF2Count());
		out.writeProperty(PersistConst.PROPERTY_WEIGHTS_F1_F2,
				bam.getWeightsF1toF2());
		out.writeProperty(PersistConst.PROPERTY_WEIGHTS_F2_F1,
				bam.getWeightsF2toF1());

		out.flush();

	}

}
