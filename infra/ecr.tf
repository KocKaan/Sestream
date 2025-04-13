resource "aws_ecr_repository" "main" {
  name = "sestream"
}

resource "aws_ecr_lifecycle_policy" "main" {
  repository = aws_ecr_repository.main.name
  policy     = <<-EOT
    {
      "rules": [
        {
          "rulePriority": 1,
          "description": "Remove untagged images",
          "selection": {
            "tagStatus": "untagged",
            "countType": "imageCountMoreThan",
            "countNumber": 1
          },
          "action": {
            "type": "expire"
          }
        }
      ]
    }
  EOT
}