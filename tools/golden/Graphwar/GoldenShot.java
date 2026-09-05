package Graphwar;

import GraphServer.Constants;

/**
 * Golden-test harness for the Graphwar Python reimplementation (M1).
 *
 * Fires one shot through the *reference* {@link Function#processFunctionRange}
 * and prints the result as JSON so the Python simulator can be compared
 * against it. This is a test-only driver; it is NOT part of the game.
 *
 * Usage:
 *   java -cp bin Graphwar.GoldenShot "<func>" <inverted> "<shooterX,shooterY>" \
 *        ["<sx,sy,alive>" ...] ["C:<cx,cy,r>" ...]
 *
 *   func      the firing function string (e.g. "x^2*0.01")
 *   inverted  0 or 1 (TEAM2 mirror)
 *   shooter   plane/pixel coords of the current-turn soldier (int, y down)
 *   soldiers  zero or more "x,y,alive" (alive 0/1); the shooter is one of them
 *   C:        zero or more terrain circles "x,y,r" (plane coords)
 *
 * Output (single JSON line):
 *   {"numSteps":N,"lastX":d,"lastY":d,
 *    "hits":[[player,soldier,position],...],
 *    "points":[[x,y],...]}
 *
 * `points` are the integrated trajectory in plane/pixel coords (muzzle through
 * the last step), with the TEAM2 mirror applied to x — exactly what the Python
 * `physics.process_function_range` returns, so the two are directly comparable.
 * `lastX`/`lastY` are the raw reference values (no mirror), matching the source.
 */
public class GoldenShot
{
	public static void main(String[] args) throws Exception
	{
		if (args.length < 3)
		{
			System.err.println("usage: GoldenShot <func> <inverted> <shooterX,shooterY> [soldier...] [C:circle...]");
			System.exit(2);
		}

		String funcStr = args[0];
		boolean inverted = args[1].trim().equals("1");
		int[] shooter = parseXY(args[2]);

		// Collect soldiers and circles.
		java.util.List<int[]> soldierSpecs = new java.util.ArrayList<>();
		java.util.List<int[]> circleSpecs = new java.util.ArrayList<>();
		for (int i = 3; i < args.length; i++)
		{
			String a = args[i].trim();
			if (a.startsWith("C:"))
			{
				circleSpecs.add(parseInts(a.substring(2)));
			}
			else
			{
				soldierSpecs.add(parseInts(a));
			}
		}

		// Build the terrain from the given circles.
		int[] circleInfo = new int[circleSpecs.size() * 3];
		for (int i = 0; i < circleSpecs.size(); i++)
		{
			int[] c = circleSpecs.get(i);
			circleInfo[3 * i] = c[0];
			circleInfo[3 * i + 1] = c[1];
			circleInfo[3 * i + 2] = c[2];
		}
		Obstacle obstacle = new Obstacle(circleSpecs.size(), circleInfo);

		// Build players. Each soldier belongs to its own single-soldier player so
		// that the (player, soldier) hit identity is unambiguous and the shooter
		// is exactly one soldier.
		int numSoldiers = soldierSpecs.size();
		Player[] players = new Player[numSoldiers];
		int shooterPlayer = -1;
		for (int i = 0; i < numSoldiers; i++)
		{
			int[] s = soldierSpecs.get(i);
			players[i] = new Player("p" + i, i, Constants.TEAM1, false, 1, false);
			players[i].startSoldier(0, s[0], s[1]);
			if (s[0] == shooter[0] && s[1] == shooter[1])
			{
				shooterPlayer = i;
			}
		}
		if (shooterPlayer < 0)
		{
			System.err.println("shooter not found among soldiers");
			System.exit(3);
		}

		Function fn = new Function(funcStr);
		fn.processFunctionRange(obstacle, players, numSoldiers, shooterPlayer, inverted);

		int numSteps = fn.getNumSteps();
		double lastX = fn.getLastX();
		double lastY = fn.getLastY();
		int numHits = fn.getNumPlayersHit();

		// Reconstruct plane-coord points exactly as the Python port does.
		StringBuilder sb = new StringBuilder();
		sb.append("{");
		sb.append("\"numSteps\":").append(numSteps).append(",");
		sb.append("\"lastX\":").append(lastX).append(",");
		sb.append("\"lastY\":").append(lastY).append(",");

		sb.append("\"hits\":[");
		for (int i = 0; i < numHits; i++)
		{
			if (i > 0) sb.append(",");
			sb.append("[").append(fn.getPlayerHit(i)).append(",")
			  .append(fn.getSoldierHit(i)).append(",")
			  .append(fn.getSoldierHitPosition(i)).append("]");
		}
		sb.append("],");

		sb.append("\"points\":[");
		for (int i = 0; i < numSteps; i++)
		{
			if (i > 0) sb.append(",");
			double x = Constants.PLANE_LENGTH * fn.getX(i) / Constants.PLANE_GAME_LENGTH + Constants.PLANE_LENGTH / 2;
			double y = -Constants.PLANE_LENGTH * fn.getY(i) / Constants.PLANE_GAME_LENGTH + Constants.PLANE_HEIGHT / 2;
			if (inverted)
			{
				x = Constants.PLANE_LENGTH - x;
			}
			sb.append("[").append(x).append(",").append(y).append("]");
		}
		sb.append("]}");

		System.out.println(sb.toString());

		// When terrain circles are present, dump the exact collidePoint grid so
		// the Python side can use byte-identical terrain (anti-aliasing makes a
		// hand reimplementation of the oval fill inexact). One line: a JSON array
		// of PLANE_HEIGHT strings, each PLANE_LENGTH chars of '0'/'1'.
		if (circleSpecs.size() > 0)
		{
			StringBuilder grid = new StringBuilder();
			grid.append("[");
			for (int py = 0; py < Constants.PLANE_HEIGHT; py++)
			{
				if (py > 0) grid.append(",");
				grid.append("\"");
				for (int px = 0; px < Constants.PLANE_LENGTH; px++)
				{
					grid.append(obstacle.collidePoint(px, py) ? '1' : '0');
				}
				grid.append("\"");
			}
			grid.append("]");
			System.out.println(grid.toString());
		}
	}

	private static int[] parseXY(String s)
	{
		String[] p = s.split(",");
		return new int[] { Integer.parseInt(p[0].trim()), Integer.parseInt(p[1].trim()) };
	}

	private static int[] parseInts(String s)
	{
		String[] p = s.split(",");
		int[] out = new int[p.length];
		for (int i = 0; i < p.length; i++)
			out[i] = Integer.parseInt(p[i].trim());
		return out;
	}
}
