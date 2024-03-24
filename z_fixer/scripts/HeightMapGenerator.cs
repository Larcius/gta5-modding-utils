using GTA;
using GTA.Math;
using GTA.Native;
using GTA.UI;
using System;
using System.IO;
using System.Windows.Forms;

namespace HeightMap
{
	public class HeightMapGenerator : Script
	{
		private const float maxCoordinate = 10000f;
		private const int maxHeight = 2048;

		// resolution (distance in meters between two samples)
		private const float resolution = 0.1f;

		private static Vector2[] unitHexagon = new Vector2[6];
		static HeightMapGenerator() {
			double angleStep = 2d * Math.PI / unitHexagon.Length;
			for (int i = 0; i < unitHexagon.Length; i++) {
				unitHexagon[i] = new Vector2((float) Math.Sin(angleStep * i), (float) Math.Cos(angleStep * i));
			}
		}

		public StreamWriter writer;
		private StreamReader reader;

		private string filenameInput = "Full path (e.g. C:\\<SOME PATH>\\z_fixer\\generated\\coords.txt)";

		public bool startRequested, abortRequested;

		private int stepSize;

		private float prevZ;

		private Vector3 coords;
		private Quaternion rotation;
		private float radius;

		public HeightMapGenerator()
		{
			Tick += OnTick;
			KeyDown += OnKeyDown;
		}

		private void OnKeyDown(object sender, KeyEventArgs e)
		{
			if (e.KeyCode != Keys.F10) return;

			if (writer == null) {
				string newFilenameInput = Game.GetUserInput(WindowTitle.EnterMessage60, filenameInput, 255);
				if (newFilenameInput == null || newFilenameInput == "") {
					return;
				}
				filenameInput = newFilenameInput;
				string filenameOutput = "hmap.txt";
				try {
					reader = new StreamReader(filenameInput);
				} catch (IOException ex) {
					Notification.Show("invalid filename");
					return;
				}
				writer = File.CreateText(filenameOutput);
				abortRequested = false;
				startRequested = true;
			} else {
				abortRequested ^= true;
			}
		}

		private bool readCoords() {
			string line;
			while ((line = reader.ReadLine()) != null) {
				string[] splitted = line.Split(new char[] { '|', ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
				if (splitted.Length == 0) {
					continue;
				}

				if (splitted.Length != 8) {
					writer.Write("ERROR: malformed line: ");
					writer.WriteLine(line);
					continue;
				}

				float x, y, z, qx, qy, qz, qw, r;
				if (!Single.TryParse(splitted[0], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out x) ||
						!Single.TryParse(splitted[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out y) ||
						!Single.TryParse(splitted[2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out z) ||
						!Single.TryParse(splitted[3], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out qx) ||
						!Single.TryParse(splitted[4], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out qy) ||
						!Single.TryParse(splitted[5], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out qz) ||
						!Single.TryParse(splitted[6], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out qw) ||
						!Single.TryParse(splitted[7], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out r)) {

					writer.Write("ERROR: malformed coordinates: ");
					writer.WriteLine(line);
					continue;
				}

				if (x <= -maxCoordinate || x >= maxCoordinate || y <= -maxCoordinate || y >= maxCoordinate) {
					writer.Write("ERROR: coordinates out of range: ");
					writer.WriteLine(line);
					continue;
				}

				if (Single.IsNaN(z) || z < 0) {
					z = 0;
				} else if (z > maxHeight) {
					z = maxHeight;
				}

				coords = new Vector3(x, y, z);
				rotation = new Quaternion(qx, qy, qz, qw);
				radius = Math.Abs(r);
				return true;
			}
			return false;
		}

		private void finish()
		{
			reader.Close();
			reader = null;

			writer.Flush();
			writer.Close();
			writer = null;
		}

		private Vector3 getSurfaceNormalAt(float x, float y, float radius) {
			Vector3 center = new Vector3(x, y, getHeightEpsHexagon(x, y));
			Vector3[] hexagon = new Vector3[unitHexagon.Length];

			for (int i = 0; i < unitHexagon.Length; i++) {
				float xh = x + unitHexagon[i].X * radius;
				float yh = y + unitHexagon[i].Y * radius;
				hexagon[i] = new Vector3(xh, yh, getHeightEpsHexagon(xh, yh));
			}

			Vector3 normal = new Vector3(0f, 0f, 0f);
			for (int i = 0; i < hexagon.Length; i++) {
				Vector3 triangleNormal = getNormalOfTriangle(hexagon[i], center, hexagon[(i + 1) % hexagon.Length]); // must be in counter-clockwise order
				normal += triangleNormal;
			}

			return Vector3.Normalize(normal);
		}

		private Vector3 getNormalOfTriangle(Vector3 a, Vector3 b, Vector3 c) {
			Vector3 normal = Vector3.Cross(b - a, c - a);
			return Vector3.Normalize(normal);
		}

		// unfortunately checking only exactly that point is not reliable because you may hit a small hole in the surface.
		// So check not only a single point but also points on an eps hexagon around it
		private float getHeightEpsHexagon(float x, float y) {
			const float eps = 0.025f;
			float result = getHeight(x, y);
			for (int i = 0; Single.IsNaN(result) && i < unitHexagon.Length; i++) {
				result = getHeight(x + unitHexagon[i].X * eps, y + unitHexagon[i].Y * eps);
			}
			return result;
		}

		private float getHeight(float x, float y) {
			float result = World.GetGroundHeight(new Vector2(x, y));
			if (result == 0) {
				return Single.NaN;
			}
			return result;
		}

		private bool isApproxEqual(float f1, float f2) {
			return Math.Abs(f1 - f2) < 0.001f;
		}

		private void setPlayerPosition(float x, float y, float z) {
			prevZ = z;
			Game.Player.Character.Position = new Vector3(x, y, z);
		}

		private Vector3 getClosestPointOnLineSegment(Vector3 pos, Vector3 linePoint1, Vector3 linePoint2) {
			Vector3 d = (linePoint2 - linePoint1).Normalized;
			Vector3 v = pos - linePoint1;
			float t = Vector3.Dot(v, d);
			return linePoint1 + d * t;
		}

		private Vector3 getNearestPositionOnStreet(Vector3 position) {
			OutputArgument outPos1 = new OutputArgument();
			OutputArgument outPos2 = new OutputArgument();

			Function.Call(Hash.GET_CLOSEST_ROAD, position.X, position.Y, position.Z, 0.0f, 0, outPos1, outPos2);
			Vector3 next1 = outPos1.GetResult<Vector3>();
			Vector3 next2 = outPos2.GetResult<Vector3>();

			return getClosestPointOnLineSegment(position, next1, next2);
		}

		private bool isPointOnRoad(float x, float y, float z) {
			return Function.Call<bool>(Hash.IS_POINT_ON_ROAD, x, y, z);
		}

		private bool isPointInWater(float x, float y, float groundHeight) {
			OutputArgument outHeight = new OutputArgument();
			bool hitWater = Function.Call<bool>(Hash.GET_WATER_HEIGHT_NO_WAVES, x, y, groundHeight, outHeight);
			float height = outHeight.GetResult<float>();
			return hitWater && groundHeight < height;
		}

		private bool areNodesLoadedAroundPoint(float x, float y, float z) {
			float offset = 25f;
			return areNodesLoadedForArea(x - offset, y - offset, z - offset, x + offset, y + offset, z + offset);
		}

		private bool areNodesLoadedForArea(float x1, float y1, float z1, float x2, float y2, float z2) {
			return Function.Call<bool>(Hash.IS_NAVMESH_LOADED_IN_AREA, x1, y1, z1, x2, y2, z2);
		}

		private void OnTick(object sender, EventArgs e)
		{
			if (writer == null) {
				return;
			}

			if (abortRequested) {
				finish();
				return;
			}

			if (startRequested) {
				startRequested = false;
			} else {
				bool addWarning;
				float z;
				if (stepSize == 0) {
					addWarning = true;
					if (Single.IsNaN(coords.Z)) {
						z = 0f;
					} else {
						z = coords.Z;
					}
				} else {
					addWarning = false;
					z = getHeightEpsHexagon(coords.X, coords.Y);

					if (Single.IsNaN(z)) {
						z = prevZ - stepSize;

						if (z <= 0) {
							stepSize /= 2;
							z = maxHeight - stepSize;
						} else if (isApproxEqual(z % (2f * stepSize), 0)) {
							z -= stepSize;
						}
					}
					if (stepSize == 0 || !isApproxEqual(prevZ, z)) {
						setPlayerPosition(coords.X, coords.Y, z);
						return;
					}
				}

				if (!areNodesLoadedAroundPoint(coords.X, coords.Y, z)) {
					return;
				}

				writer.Write(coords.X.ToString(System.Globalization.CultureInfo.InvariantCulture));
				writer.Write(",");
				writer.Write(coords.Y.ToString(System.Globalization.CultureInfo.InvariantCulture));
				writer.Write(",");
				writer.Write(z.ToString(System.Globalization.CultureInfo.InvariantCulture));
				writer.Write(",");

				float minZ = z;
				bool isOnRoad = isPointOnRoad(coords.X, coords.Y, z);
				bool isInWater = isPointInWater(coords.X, coords.Y, z);
				bool finalLoop = false;
				for (float curRadius = resolution; radius > 0 && !finalLoop; curRadius += resolution) {
					if (curRadius >= radius + resolution - 0.001f) {
						finalLoop = true;
						curRadius = radius + Math.Min(Math.Max(0.5f, radius), 1f);
					} else if (curRadius >= radius - 0.001f) {
						// if curRadius is approx. or even greater than the max radius then set it to max radius
						curRadius = radius;
					}

					// calculate number of angle steps depending on the radius
					int numAngleSteps;
					if (resolution / curRadius > Math.Sqrt(3)) {
						// use at least 3 angle steps
						numAngleSteps = 3;
					} else {
						// using the formula for side-length of an regular n-polygon for a given outer radius:
						// side-length = 2 * outer-radius * sin(PI / n)
						// note that the previous check also ensured that the argument for arcsin is always less than 1
						// furthermore it is ensured that the result of arcsin is always strictly positive
						// since resolution and curRadius are always strictly positive
						numAngleSteps = (int) Math.Ceiling(Math.PI / (Math.Asin(resolution / curRadius / 2f)));
					}

					double angleStep = 2d * Math.PI / numAngleSteps;
					for (double angle = 0; angle < 2d * Math.PI; angle += angleStep) {
						Vector3 point = new Vector3((float) Math.Sin(angle) * curRadius, (float) Math.Cos(angle) * curRadius, 0f);
						point = rotation * point;

						float curX = coords.X + point.X;
						float curY = coords.Y + point.Y;
						float curZ = getHeightEpsHexagon(curX, curY);

						if (Single.IsNaN(curZ)) {
							continue;
						}

						if (finalLoop) {
							// it is sufficient to check only the center and the circumcircle
							if (!isOnRoad) {
								isOnRoad = isPointOnRoad(curX, curY, curZ);
							}
							if (!isInWater) {
								isInWater = isPointInWater(curX, curY, curZ);
							}
						} else {
							minZ = Math.Min(minZ, curZ - point.Z);
						}
					}
				}

				writer.Write(minZ.ToString(System.Globalization.CultureInfo.InvariantCulture));

				Vector3 normal = getSurfaceNormalAt(coords.X, coords.Y, radius * 2f);
				writer.Write(",");
				writer.Write(normal.X.ToString(System.Globalization.CultureInfo.InvariantCulture));
				writer.Write(",");
				writer.Write(normal.Y.ToString(System.Globalization.CultureInfo.InvariantCulture));
				writer.Write(",");
				writer.Write(normal.Z.ToString(System.Globalization.CultureInfo.InvariantCulture));

				Vector3 position = new Vector3(coords.X, coords.Y, z);
				Vector3 nextPositionOnStreet = getNearestPositionOnStreet(position);
				float distanceToStreet = position.DistanceTo(nextPositionOnStreet);
				writer.Write(",");
				writer.Write(distanceToStreet.ToString(System.Globalization.CultureInfo.InvariantCulture));

				writer.Write(",");
				writer.Write(isOnRoad);
				writer.Write(",");
				writer.Write(isInWater);

				if (addWarning) {
					writer.Write(",WARNING: could not get z coordinate so just used the one from the input file or 0 if NaN was given");
				}

				writer.WriteLine("");
			}

			if (readCoords()) {
				// prepare for next step
				stepSize = 2 * maxHeight;
				setPlayerPosition(coords.X, coords.Y, coords.Z);
			} else {
				finish();
			}
		}
	}
}
