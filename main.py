import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# Lưu trữ hệ số của các player
coefficients_list = []

for player in X.columns:
    # Chỉ lấy các trận đấu mà player tham gia
    player_data = df[df[player] != 0]
    
    if not player_data.empty:
        # Tách dữ liệu đầu vào và nhãn cho player
        X_player = player_data.drop(columns=['Result'])
        y_player = player_data['Result']
        
        # Huấn luyện mô hình logistic regression cho player
        model = LogisticRegression()
        model.fit(X_player, y_player)
        
        # Lấy hệ số của player
        coefficient = model.coef_[0][X_player.columns.get_loc(player)]
        coefficients_list.append((player, coefficient))

# Tạo DataFrame từ danh sách hệ số và sắp xếp từ cao đến thấp
coefficients_df = pd.DataFrame(coefficients_list, columns=['Player', 'Coefficient'])

@app.route('/players', methods=['GET'])
def get_player_names():
    try:
        # Read data from GitHub
        player_names = X.columns.tolist()
        return jsonify({'players': player_names}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/create_teams', methods=['POST'])
def create_teams():
    selected_players = request.json['players']
    
    # Lấy hệ số của những người chơi được chỉ định
    selected_coefficients = coefficients_df[coefficients_df['Player'].isin(selected_players)]
    
    # Tạo tất cả các tổ hợp có thể của 5 người từ danh sách 10 người
    combinations_5 = list(combinations(selected_coefficients['Player'], 5))
    
    # Tính tổng hệ số của mỗi tổ hợp
    combination_scores = []
    for comb in combinations_5:
        score = selected_coefficients[selected_coefficients['Player'].isin(comb)]['Coefficient'].sum()
        combination_scores.append((comb, score))
    
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
        new_row[player] = 1

    # Set 2 for players in team2
    for player in team2:
        new_row[player] = 2

    # Set the result (1 for team1 win, 2 for team2 win)
    new_row['Result'] = 1 if winner == 'team1' else 2

    # Append the new row to the dataframe
    df = df.append(new_row, ignore_index=True)

    # Save updated dataframe back to GitHub
    save_csv_to_github(df)

    return jsonify({"message": "Match result updated successfully"}), 200

@app.route('/sorted_coefficients', methods=['GET'])
def get_sorted_coefficients():
    # Sort the coefficients DataFrame by Coefficient in descending order
    sorted_coeffs = coefficients_df.sort_values('Coefficient', ascending=False)
    
    # Convert to a list of dictionaries for JSON serialization
    coeffs_list = sorted_coeffs.to_dict('records')
    
    return jsonify(coeffs_list)

@app.route('/')
def serve_frontend():
    return send_from_directory(os.getcwd(), 'index.html')
@app.route('/player')
def index2():
    return send_from_directory(os.getcwd(), 'index2.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 8001)))
