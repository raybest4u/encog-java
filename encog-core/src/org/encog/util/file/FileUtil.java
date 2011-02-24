package org.encog.util.file;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;

import org.encog.EncogError;
import org.encog.bot.BotUtil;

public class FileUtil {

	public static File addFilenameBase(File filename, String base) {
		String f = getFileName(filename);
		String ext = getFileExt(filename);

		int idx1 = f.lastIndexOf('_');
		int idx2 = f.lastIndexOf(File.separatorChar);

		boolean remove = false;

		if (idx1 != -1) {
			if (idx2 == -1) {
				remove = true;
			} else {
				remove = idx1 > idx2;
			}
		}

		if (remove) {
			f = f.substring(0, idx1);
		}

		return new File(f + base + "." + ext);
	}

	public static String getFileName(File file) {
		String fileName = file.toString();
		int mid = fileName.lastIndexOf(".");
		if (mid == -1) {
			return fileName;
		}
		return fileName.substring(0, mid);
	}

	public static String getFileExt(File file) {
		String fileName = file.toString();
		int mid = fileName.lastIndexOf(".");
		if (mid == -1)
			return "";
		return fileName.substring(mid + 1, fileName.length());
	}

	public static String readFileAsString(File filePath)
			throws java.io.IOException {
		StringBuffer fileData = new StringBuffer(1000);
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		char[] buf = new char[1024];
		int numRead = 0;
		while ((numRead = reader.read(buf)) != -1) {
			String readData = String.valueOf(buf, 0, numRead);
			fileData.append(readData);
			buf = new char[1024];
		}
		reader.close();
		return fileData.toString();
	}

	public static String forceExtension(String name, String ext) {
		String b = getFileName(new File(name));
		return b + "." + ext;
	}

	public static void writeFileAsString(File path, String str)
			throws IOException {

		BufferedWriter o = new BufferedWriter(new FileWriter(path));
		o.write(str);
		o.close();
	}

	public static void copy(File source, File target) {
		try {
			final byte[] buffer = new byte[BotUtil.BUFFER_SIZE];

			int length;

			final FileOutputStream fos = new FileOutputStream(target);
			final InputStream is = new FileInputStream(source);

			do {
				length = is.read(buffer);

				if (length >= 0) {
					fos.write(buffer, 0, length);
				}
			} while (length >= 0);

			fos.close();
		} catch (final IOException e) {
			throw new EncogError(e);
		}

	}

}
