package Graphwar;
public class ProbeTerrain {
  public static void main(String[] a) {
    int[] empty = new int[0];
    Obstacle o = new Obstacle(0, empty);
    System.out.println("white_center(385,225) collide=" + o.collidePoint(385,225));
    System.out.println("white(10,10) collide=" + o.collidePoint(10,10));
    System.out.println("oob(-1,10) collide=" + o.collidePoint(-1,10));
    System.out.println("oob(800,10) collide=" + o.collidePoint(800,10));
    int[] circ = {385,225,20};
    Obstacle c = new Obstacle(1, circ);
    System.out.println("black_center(385,225) collide=" + c.collidePoint(385,225));
    System.out.println("black_edge(385,205) collide=" + c.collidePoint(385,205));
    System.out.println("black_just_out(385,203) collide=" + c.collidePoint(385,203));
    System.out.println("white_far(10,10) collide=" + c.collidePoint(10,10));
  }
}
