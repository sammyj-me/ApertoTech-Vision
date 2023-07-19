import video_processing

def main():
    path = video_processing.open_file_dialog()
    file_to_analyze = video_processing.Analyzer(path)
    file_to_analyze.analyze_file()

if __name__ == "__main__":
    main()