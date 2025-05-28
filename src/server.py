#
# Web based GUI for Chess Engine
#
# packages
from flask import Flask
from flask import request
from flask import send_from_directory
import chess
import io
import json
from flask import jsonify
from flask import Response
from flask_cors import CORS
from datetime import datetime
import os

# Import your engine libraries
from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine as engine_lib

# create web app instance
app = Flask(__name__, 
           static_folder='uci-gui/src/static',
           static_url_path='/static')
CORS(app)  # Enable CORS for cross-origin requests

# Initialize your engine (no global game state needed)
engine = constants.ENGINE_BUILDERS['local']() 

@app.route('/')
def serve_index():
    return send_from_directory('uci-gui/src', 'index.html')

@app.route('/api/get_engine_move', methods=['POST'])
def get_engine_move():
    """Get engine move for given position"""
    data = request.get_json()
    fen = data.get('fen')
    
    if not fen:
        return jsonify({'success': False, 'error': 'No FEN provided'})
    
    try:
        # Create board from FEN
        board = chess.Board(fen)
        
        # Check if game is already over
        if board.is_game_over():
            return jsonify({
                'success': False, 
                'error': 'Game is over',
                'game_over': True,
                'result': str(board.outcome())
            })
        
        # Use your engine to get the best move
        predicted_move = engine.play(board=board).uci()
        engine_move = chess.Move.from_uci(predicted_move)
        
        if engine_move in board.legal_moves:
            # Make the engine move
            board.push(engine_move)
            
            return jsonify({
                'success': True,
                'move': predicted_move,
                'fen': board.fen(),
                'game_over': board.is_game_over(),
                'result': str(board.outcome()) if board.is_game_over() else None
            })
        else:
            return jsonify({'success': False, 'error': 'Engine returned illegal move'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Engine error: {str(e)}'})

@app.route('/api/validate_move', methods=['POST'])
def validate_move():
    """Validate a move for given position"""
    data = request.get_json()
    fen = data.get('fen')
    move_uci = data.get('move')
    
    if not fen or not move_uci:
        return jsonify({'success': False, 'error': 'FEN and move required'})
    
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        
        if move in board.legal_moves:
            # Make the move
            board.push(move)
            return jsonify({
                'success': True,
                'fen': board.fen(),
                'legal': True,
                'game_over': board.is_game_over(),
                'result': str(board.outcome()) if board.is_game_over() else None
            })
        else:
            return jsonify({
                'success': False, 
                'legal': False,
                'error': 'Illegal move'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Invalid move or FEN: {str(e)}'})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.before_request
def log_request_info():
    print(f"🔍 Request: {request.method} {request.path}")
    if request.args:
        print(f"   Args: {dict(request.args)}")

@app.after_request
def log_response_info(response):
    print(f"📤 Response: {response.status_code} for {request.path}")
    return response

# main driver
if __name__ == '__main__':
    # Verify the file exists before starting
    index_path = os.path.join('uci-gui', 'src', 'index.html')
    if not os.path.exists(index_path):
        print(f"⚠️  Warning: {index_path} not found!")
        print("   Make sure your directory structure is:")
        print("   📁 your_server_directory/")
        print("   ├── 📄 server.py")
        print("   └── 📁 uci-gui/")
        print("       └── 📁 src/")
        print("           └── 📄 index.html")
    else:
        print(f"✅ Found index.html at {index_path}")
    
    print("🚀 Starting Chess Engine Server")
    print("🌐 Access your chess GUI at: http://localhost:8080")
    print("🔗 API endpoints available at: http://localhost:8080/api/")
    
    # start HTTP server on port 8080
    app.run(debug=True, threaded=True, host='0.0.0.0', port=8080)