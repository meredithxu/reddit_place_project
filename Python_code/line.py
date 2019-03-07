

class Line:
  """
    Line objects represent each straight edge of a Path
    The Lines consist of two points, the start and end points
  """

  def __init__(self, point1, point2):
    self.point1 = point1
    self.point2 = point2

  def set_point_1(self, x, y):
    self.point1 = Point(x,y)

  def set_point_2(self, x, y):
    self.point1 = Point(x,y)

  def get_point1(self):
    return self.point1

  def get_point2(self):
    return self.point2

  def point_is_on_leftside_of_line(self, point):
    """
      Given coordinates (x,y), checks whether drawing a horizontal line rightward from the point will cross this Line object
      Return True if so and False if not
    """

    if (point.y < self.point2.y and point.y >= self.point1.y) or (point.y >= self.point2.y and point.y < self.point1.y):

      # The equation to determine this comes from here: https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located

      d = ((point.x - self.point1.x)*(self.point2.y - self.point1.y)) - ((point.y - self.point1.y)*(self.point2.x - self.point1.x))

      left = (self.point1.y - self.point2.y)
      # If d and left are both positive or both negative, then the point is on the left side of the line

      if (d < 0 and left < 0) or (d > 0 and left > 0):
        return True

      else:
        return False

  def point_is_on_the_line(self, point):
    """
      Given coordinates (x,y), checks whether the point is over the line segment.
      Return True if so and False if not
    """
    if (point.y <= self.point2.y and point.y >= self.point1.y) or (point.y >= self.point2.y and point.y <= self.point1.y):
      if (point.x <= self.point2.x and point.x >= self.point1.x) or (point.x >= self.point2.x and point.x <= self.point1.x):
        d = ((point.x - self.point1.x)*(self.point2.y - self.point1.y)) - ((point.y - self.point1.y)*(self.point2.x - self.point1.x))
        
        if d == 0.:
          return True

    return False

