
class Path:

  def __init__(self, pic_id):
    self.lines = []
    self.pic_id = pic_id
    self.left = None
    self.right = None
    self.bottom = None
    self.top = None


  def add_line(self, line):
    """
      Precondition: line is a Line object
      Adds the line to the end of Path
    """
    self.lines.append(line)


    if self.left == None or line.get_point1().get_x() < self.left:
      self.left = line.get_point1().get_x()
    if self.left == None or line.get_point2().get_x() < self.left:
      self.left = line.get_point2().get_x()

    if self.right == None or line.get_point1().get_x() > self.right:
      self.right = line.get_point1().get_x()
    if self.right == None or line.get_point2().get_x() > self.right:
      self.right = line.get_point2().get_x()

    if self.bottom == None or line.get_point1().get_y() < self.bottom:
      self.bottom = line.get_point1().get_y()
    if self.bottom == None or line.get_point2().get_y() < self.bottom:
      self.bottom = line.get_point2().get_y()
      
    if self.top == None or line.get_point1().get_y() > self.top:
      self.top = line.get_point1().get_y()
    if self.top == None or line.get_point2().get_y() > self.top:
      self.top = line.get_point2().get_y()




  def get_line(self, index):
    """
      Return the line along the path that is stored at index in self.lines
    """
    return self.lines[index]

  def get_all_lines(self):
    """
      Return all the lines along the path
    """
    return self.lines

  def get_top(self):
    return self.top

  def get_bottom(self):
    return self.bottom

  def get_left(self):
    return self.left

  def get_right(self):
    return self.right

  def pixel_is_in_image(self, point):
    """
        Returns true if pixel is in image enclosed by self.path

        path is a list of the path coordinates of the edge

    """

    # Do horizontal line test
    # If line passes through odd number of edges, it is inside the image
    # If line passes through even number of edges, it is outside the image
    num_intersections = 0
    
    for line in self.lines:
      if (line.point_is_on_leftside_of_line(point)):
        num_intersections += 1

    return (num_intersections % 2 == 1)