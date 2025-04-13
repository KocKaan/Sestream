resource "aws_eks_node_group" "main" {
  node_group_name = "sestream"
  version         = "1.32"
  cluster_name    = "shindig"
  node_role_arn   = "arn:aws:iam::392223633930:role/node"
  subnet_ids      = ["subnet-0065cc608c9d24dfb", "subnet-0282d77542c8d0944"]
  instance_types  = ["t2.micro"]

  scaling_config {
    desired_size = 2
    max_size     = 2
    min_size     = 2
  }
}