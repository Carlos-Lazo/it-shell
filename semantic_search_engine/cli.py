import argparse
from pathlib import Path
from semantic_search_engine.bootstrap import create_engine
from semantic_search_engine.shared.contracts import QueryRequest

def main():
    parser=argparse.ArgumentParser(description='Propuestas de comandos; no ejecuta shell')
    parser.add_argument('query'); parser.add_argument('--config',type=Path,default=Path('semantic_search_engine/config/default.yaml'))
    parser.add_argument('--format',choices=('json','markdown','cli'),default='cli'); parser.add_argument('--top-k',type=int,default=50)
    args=parser.parse_args()
    print(create_engine(args.config).search(QueryRequest(args.query,args.top_k,args.format)))
if __name__=='__main__': main()
