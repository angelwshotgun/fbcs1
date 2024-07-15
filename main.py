import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from github import Github
import os
from io import StringIO
from itertools import combinations
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory


load_dotenv()

app = Flask(__name__)

# GitHub configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_NAME = 'angelwshotgun/fbcs' 
FILE_PATH = 'match_data.csv'

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

def read_csv_from_github():
    file_content = repo.get_contents(FILE_PATH)
    file_data = file_content.decoded_content.decode('utf-8')
    return pd.read_csv(StringIO(file_data))

def save_csv_to_github(data):
    csv_content = data.to_csv(index=False)
    file = repo.get_contents(FILE_PATH)
    repo.update_file(FILE_PATH, "Update CSV file", csv_content, file.sha)

# Read data from GitHub
data = read_csv_from_github()

df = pd.DataFrame(data)

# Tách dữ liệu đầu vào và nhãn
X = df.drop(columns=['Result'])
y = df['Result']

np.random.seed(42)

coefficients_list = []

for player in X.columns:
    # Chỉ lấy các trận đấu mà player tham gia
    player_data = df[df[player] != 0]
    
    if not player_data.empty:
        # Tách dữ liệu đầu vào và nhãn cho player
        X_player = player_data.drop(columns=['Result'])
        y_player = player_data['Result']
        
        # Huấn luyện mô hình logistic regression cho player với seed cố định
        model = LogisticRegression(random_state=42)
        model.fit(X_player, y_player)
        
        # Lấy hệ số của player
        coefficient = model.coef_[0][X_player.columns.get_loc(player)]
        coefficients_list.append((player, coefficient))

# Tạo DataFrame từ danh sách hệ số và sắp xếp từ cao đến thấp
coefficients_df = pd.DataFrame(coefficients_list, columns=['Player', 'Coefficient'])

def calculate_pair_coefficients(players):
    pair_coefficients = {}
    for p1, p2 in combinations(players, 2):
            matches = df[(df[p1] != 0) & (df[p2] != 0)]
            if not matches.empty:
                # Tính tỷ lệ thắng khi hai người chơi cùng đội
                win_rate = matches[(matches[p1] == matches[p2]) & (matches['Result'] == matches[p1])].shape[0] / matches.shape[0]
                # Chuẩn hóa hệ số kết hợp
                pair_coefficients[(p1, p2)] = (win_rate - 0.5) * 2
    return pair_coefficients

def retrain_model():
    global coefficients_df, X, y
    
    # Read the latest data
    data = read_csv_from_github()
    df = pd.DataFrame(data)
    
    # Update X and y
    X = df.drop(columns=['Result'])
    y = df['Result']
    
    coefficients_list = []
    
    for player in X.columns:
        player_data = df[df[player] != 0]
        
        if not player_data.empty:
            X_player = player_data.drop(columns=['Result'])
            y_player = player_data['Result']
            
            model = LogisticRegression(random_state=42)
            model.fit(X_player, y_player)
            
            coefficient = model.coef_[0][X_player.columns.get_loc(player)]
            coefficients_list.append((player, coefficient))
    
    # Update coefficients_df
    coefficients_df = pd.DataFrame(coefficients_list, columns=['Player', 'Coefficient'])

@app.route('/players', methods=['GET'])
def get_player_names():
    try:
        # Read data from GitHub
        player_names = X.columns.tolist()
        return jsonify({'players': player_names}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
from itertools import combinations

@app.route('/create_teams', methods=['POST'])
def create_teams():
    selected_players = request.json['players']
    
    # Lấy hệ số của những người chơi được chỉ định
    selected_coefficients = coefficients_df[coefficients_df['Player'].isin(selected_players)]
    
    # Tính toán hệ số kết hợp cho mỗi cặp người chơi
    pair_coefficients = calculate_pair_coefficients(selected_players)
    
    # Tạo tất cả các tổ hợp có thể của 5 người từ danh sách 10 người
    combinations_5 = list(combinations(selected_coefficients['Player'], 5))
    
    # Tính tổng hệ số của mỗi tổ hợp, bao gồm cả hệ số kết hợp
    combination_scores = []
    for comb in combinations_5:
        individual_score = selected_coefficients[selected_coefficients['Player'].isin(comb)]['Coefficient'].sum()
        pair_score = sum(pair_coefficients.get((p1, p2), 0) for p1, p2 in combinations(comb, 2))
        total_score = individual_score + pair_score
        combination_scores.append((comb, total_score))
    
    # Tạo DataFrame từ danh sách tổ hợp và tổng hệ số
    combination_scores_df = pd.DataFrame(combination_scores, columns=['Combination', 'Total_Coefficient'])
    
    # Tìm hai tổ hợp có tổng hệ số gần nhất mà không trùng lặp người chơi
    min_diff = float('inf')
    best_comb_1 = None
    best_comb_2 = None
    
    for i in range(len(combination_scores_df)):
        for j in range(i+1, len(combination_scores_df)):
            comb1 = set(combination_scores_df.iloc[i]['Combination'])
            comb2 = set(combination_scores_df.iloc[j]['Combination'])
            if not comb1.intersection(comb2):  # Kiểm tra xem hai tổ hợp có trùng người chơi không
                diff = abs(combination_scores_df.iloc[i]['Total_Coefficient'] - combination_scores_df.iloc[j]['Total_Coefficient'])
                if diff < min_diff:
                    min_diff = diff
                    best_comb_1 = combination_scores_df.iloc[i]
                    best_comb_2 = combination_scores_df.iloc[j]
    
    # Kiểm tra xem có tìm được tổ hợp phù hợp không
    if best_comb_1 is None or best_comb_2 is None:
        return jsonify({
            "error": "Unable to create balanced teams with the selected players. Please try a different selection."
        }), 400
    
    # Trả về kết quả
    result = {
        "team1": list(best_comb_1['Combination']),
        "team2": list(best_comb_2['Combination']),
        "team1_score": float(best_comb_1['Total_Coefficient']),
        "team2_score": float(best_comb_2['Total_Coefficient'])
    }
    return jsonify(result)

@app.route('/update_match_result', methods=['POST'])
def update_match_result():
    try:
        data = request.json
        team1 = data['team1']
        team2 = data['team2']
        winner = data['winner']

        # Read current data
        df = read_csv_from_github()

        # Create a new row for this match
        new_row = pd.DataFrame(columns=df.columns)
        new_row.loc[0] = 0  # Initialize with zeros

        # Set 1 for players in team1
        for player in team1:
            if player in new_row.columns:
                new_row.at[0, player] = 1
            else:
                return jsonify({"error": f"Player {player} not found in the dataset"}), 400

        # Set 2 for players in team2
        for player in team2:
            if player in new_row.columns:
                new_row.at[0, player] = 2
            else:
                return jsonify({"error": f"Player {player} not found in the dataset"}), 400

        # Set the result (1 for team1 win, 2 for team2 win)
        if 'Result' in new_row.columns:
            new_row.at[0, 'Result'] = 1 if winner == 'team1' else 2
        else:
            return jsonify({"error": "Result column not found in the dataset"}), 400

        # Concatenate the new row to the dataframe
        df = pd.concat([df, new_row], ignore_index=True)

        # Save updated dataframe back to GitHub
        save_csv_to_github(df)

        # Retrain the model
        retrain_model()

        return jsonify({"message": "Match result updated and model retrained successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error in update_match_result: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/sorted_coefficients', methods=['GET'])
def get_sorted_coefficients():
    # Sort the coefficients DataFrame by Coefficient in descending order
    sorted_coeffs = coefficients_df.sort_values('Coefficient', ascending=False)
    
    # Convert to a list of dictionaries for JSON serialization
    coeffs_list = sorted_coeffs.to_dict('records')
    
    return jsonify(coeffs_list)

@app.route('/pair_coefficients', methods=['GET'])
def get_pair_coefficients():
    players = X.columns.tolist()
    pair_coeffs = calculate_pair_coefficients(players)
    
    # Convert to a list of dictionaries and sort by coefficient in descending order
    pair_coeffs_list = [{"player1": p1, "player2": p2, "coefficient": coeff} 
                        for (p1, p2), coeff in pair_coeffs.items()]
    pair_coeffs_list.sort(key=lambda x: x['coefficient'], reverse=True)
    
    return jsonify(pair_coeffs_list)

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        retrain_model()
        return jsonify({"message": "Model retrained successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error in retraining: {str(e)}")
        return jsonify({"error": f"An error occurred during retraining: {str(e)}"}), 500
    
@app.route('/')
def serve_frontend():
    return send_from_directory(os.getcwd(), 'index.html')
@app.route('/player')
def index2():
    return send_from_directory(os.getcwd(), 'index2.html')
@app.route('/pair')
def index3():
    return send_from_directory(os.getcwd(), 'index3.html')

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 8001)))