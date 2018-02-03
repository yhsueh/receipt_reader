#include "ros/ros.h"
#include "receipt_reader/Confirm.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "confirm_client");

  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<receipt_reader::Confirm>("confirm");
  receipt_reader::Confirm srv;
  srv.request.input = true;
  if (client.call(srv)) {
  	ROS_INFO("HI");
  }
  else {
  	ROS_INFO("HI2");
  }

  return 0;
}