import Model.model as model
import Model.t2_mandani_inference as t2_mandani_inference
import Model.t2_plot as t2_plot
from flask import Flask, jsonify, abort, request, render_template
import dataBase.db_access as db_access

app = Flask(__name__)

t2_mandani_inference.preprocessing(model.input_lvs, model.output_lv)

'''crisp = [23, 55, 45]
result = t2_mandani_inference.process(model.input_lvs, model.output_lv, model.rule_base, crisp)
print(result)'''

'''
for lv in model.input_lvs:
    t2_plot.draw_lv(lv)
t2_plot.draw_lv(model.output_lv)'''

@app.route('/api/calculate-skill-level/<java_skill>/<english_skill>/<soft_skill>', methods=['GET'])
def calculate_skill_level(java_skill, english_skill, soft_skill):
    try:
        java_skill = int(java_skill)
        english_skill = int(english_skill)
        soft_skill = int(soft_skill)
    except ValueError:
        return jsonify({"error": "all parameters must be integers"}), 400

    # Перевіряємо, чи параметри у відповідних діапазонах
    if java_skill < 0 or java_skill > 25:
        return jsonify({"error": "java_skill is out of the range [0;25]"}), 400
    if english_skill < 0 or english_skill > 60:
        return jsonify({"error": "english_skill is out of the range [0;60]"}), 400
    if soft_skill < 0 or soft_skill > 50:
        return jsonify({"error": "soft_skill is out of the range [0;50]"}), 400

    crisp = [java_skill, english_skill, soft_skill]
    result = t2_mandani_inference.process(model.input_lvs, model.output_lv, model.rule_base, crisp)
    return jsonify(result), 200


# Взаємодія з БД
@app.route('/api/register', methods=['POST'])
def save_user():
    data = request.get_json()
    username = data.get('username')
    login = data.get('login')
    password = data.get('password')

    if not username or not login or not password:
        abort(400, description="Missing fields")

    db_access.save_user(username, login, password)
    return jsonify("User saved successfully"), 200


@app.route('/api/get-user-by-login/<login>', methods=['GET'])
def get_user_by_login(login):
    user = db_access.get_user_by_login(login)
    if not user:
        abort(404, description="User not found")
    return jsonify({"user_id": user[0], "username": user[1], "password": user[2]}), 200

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    login = data.get('login')
    password = data.get('password')

    if not login or not password:
        return jsonify({"error": "Missing fields"}), 400

    user = db_access.get_user_by_login(login)
    if user[1] and user[2] == password:  # Перевірка пароля користувача
        return render_template('test.html')
    else:
        return jsonify({"error": "Invalid login or password"}), 401

@app.route('/api/save-user-result', methods=['POST'])
def save_user_result():
    data = request.get_json()
    user_id = data.get('user_id')
    java_skill = data.get('java_skill')
    english_skill = data.get('english_skill')
    soft_skill = data.get('soft_skill')
    result_number = data.get('result_number')
    result_word = data.get('result_word')

    if not all([user_id, java_skill, english_skill, soft_skill, result_number, result_word]):
        abort(400, description="Missing fields")

    db_access.save_user_result(user_id, java_skill, english_skill, soft_skill, result_number, result_word)
    return jsonify("Result saved successfully"), 200


@app.route('/api/get-user-results/<user_id>', methods=['GET'])
def get_user_results(user_id):
    results = db_access.get_user_results(user_id)
    if results:
        return jsonify(results), 200
    else:
        return jsonify({"error": "Results not found"}), 404

@app.route("/test")
def test():
    return render_template("test.html")

@app.route("/")
def index():
    return render_template("login.html")

if __name__ == '__main__':
    app.run(debug=True)


