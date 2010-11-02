package org.encog.util.cl;

/**
 * Common data held by OpenCL devices and platforms.
 */
public class EncogCLItem {

	/**
	 * Is this device or platform enabled. Disabling a platform will cause its
	 * devices to not be used either, regardless of their enabled/disabled
	 * status.
	 */
	private boolean enabled;

	/**
	 * The name of this device or platform.
	 */
	private String name;

	/**
	 * The vendor of this device or platform.
	 */
	private String vender;

	/**
	 * @return The name of this platform or device.
	 */
	public String getName() {
		return this.name;
	}

	/**
	 * @return The vender for this platform or device.
	 */
	public String getVender() {
		return this.vender;
	}

	/**
	 * @return True if this device or platform is enabled.
	 */
	public boolean isEnabled() {
		return this.enabled;
	}

	/**
	 * Enable or disable this device or platform.
	 * 
	 * @param enabled
	 *            True, if enabled.
	 */
	public void setEnabled(final boolean enabled) {
		this.enabled = enabled;
	}

	/**
	 * Set the name of this platform or device.
	 * 
	 * @param name
	 *            The name of this platform or device.
	 */
	public void setName(final String name) {
		this.name = name;
	}

	/**
	 * Set the vender for this platform or device.
	 * 
	 * @param vender
	 *            The vender.
	 */
	public void setVender(final String vender) {
		this.vender = vender;
	}

}