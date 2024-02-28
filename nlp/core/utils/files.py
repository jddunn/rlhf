import os


class Files:
    @staticmethod
    def create_models_dir(models_dir: str):
        if not os.path.exists(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "/".join(models_dir.split("/")[:-1]),
                )
            )
        ):
            os.makedirs(
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "/".join(models_dir.split("/")[:-1]),
                    )
                )
            )
        if not os.path.exists(
            os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), models_dir)
            )
        ):
            os.makedirs(
                os.path.abspath(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), models_dir)
                )
            )
        return

    @staticmethod
    def create_data_dir(path: str) -> None:
        # If path doesn't exist, make it
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def get_path(path):
        return os.path.join(os.path.dirname(__file__), path)

    @staticmethod
    def get_file(path):
        return open(Files.get_path(path), "r")

    @staticmethod
    def get_file_content(path):
        return Files.get_file(path).read()

    @staticmethod
    def get_file_lines(path):
        return Files.get_file_content(path).split("\n")

    @staticmethod
    def get_file_lines_without_empty_lines(path):
        return list(filter(lambda line: line != "", Files.get_file_lines(path)))

    @staticmethod
    def get_file_lines_without_empty_lines_and_comments(path):
        return list(
            filter(
                lambda line: line != "" and not line.startswith("#"),
                Files.get_file_lines(path),
            )
        )

    @staticmethod
    def get_file_lines_without_empty_lines_and_comments_and_spaces(path):
        return list(
            map(
                lambda line: line.strip(),
                Files.get_file_lines_without_empty_lines_and_comments(path),
            )
        )

    @staticmethod
    def get_file_lines_without_empty_lines_and_comments_and_spaces_and_new_lines(path):
        return list(
            map(
                lambda line: line.strip(),
                Files.get_file_lines_without_empty_lines_and_comments(path),
            )
        )

    @staticmethod
    def get_file_lines_without_empty_lines_and_comments_and_spaces_and_new_lines_and_tabs(
        path,
    ):
        return list(
            map(
                lambda line: line.strip(),
                Files.get_file_lines_without_empty_lines_and_comments(path),
            )
        )

    @staticmethod
    def get_file_lines_without_empty_lines_and_comments_and_spaces_and_new_lines_and_tabs_and_carriage_returns(
        path,
    ):
        return list(
            map(
                lambda line: line.strip(),
                Files.get_file_lines_without_empty_lines_and_comments(path),
            )
        )

    @staticmethod
    def get_file_lines_without_empty_lines_and_comments_and_spaces_and_new_lines_and_tabs_and_carriage_returns_and_backspaces(
        path,
    ):
        return list(
            map(
                lambda line: line.strip(),
                Files.get_file_lines_without_empty_lines_and_comments(path),
            )
        )

    @staticmethod
    def get_file_lines_without_empty_lines_and_comments_and_spaces_and_new_lines_and_tabs_and_carriage_returns_and_backspaces_and_form_feeds(
        path,
    ):
        return list(
            map(
                lambda line: line.strip(),
                Files.get_file_lines_without_empty_lines_and_comments(path),
            )
        )

    @staticmethod
    def get_file_lines_without_empty_lines_and_comments_and_spaces_and_new_lines_and_tabs_and_carriage_returns_and_backspaces_and_form_feeds_and_vertical_tabs(
        path,
    ):
        return list(
            map(
                lambda line: line.strip(),
                Files.get_file_lines_without_empty_lines_and_comments(path),
            )
        )
