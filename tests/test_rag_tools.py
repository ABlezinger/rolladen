import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'src' / 'rag' / 'tools.py'

spec = importlib.util.spec_from_file_location('rag_tools', MODULE_PATH)
rag_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_tools)


class RagToolsDateFormattingTest(unittest.TestCase):
    def test_get_list_of_available_docs_formats_numeric_dates(self):
        doc_info = [{
            'title': 'Testdokument',
            'valid_from': 20220101,
            'valid_to': 20221231,
        }]

        result = rag_tools.get_list_of_available_docs(doc_info)

        self.assertIn('gültig von 01.01.2022 bis 31.12.2022', result)


if __name__ == '__main__':
    unittest.main()
