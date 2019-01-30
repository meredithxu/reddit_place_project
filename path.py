
class Path:

  def __init__(self, pic_id):
    self.lines = []
    self.pic_id = pic_id


  def add_line(self, line):
    """
      Precondition: line is a Line object
      Adds the line to the end of Path
    """
    self.lines.append(line)



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