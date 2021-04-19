
# include <Siv3D.hpp> // OpenSiv3D v0.4.3

using namespace std;

int TABLESIZE = 30;

class snakegame {
public:
	int Heal;//餌を取得した際に回復する体力
	int HP;
	vector<vector<int>> table;//盤面の状態を保存しておくやつ
	int snake_x;//蛇の座標
	int snake_y;
	int esa_x;//餌の座標
	int esa_y;
	int tablesize;
	int score;//得点(生きながらえたターン)
	deque<pair<int, int>> position;//蛇の体の座標

	void generate() {
		//取れば得点になる餌を生成する
		int y;
		int x;
		while (1) {
			y = Random(tablesize - 1);
			x = Random(tablesize - 1);
			if (x == snake_x && y == snake_y)continue;
			break;
		}
		table[y][x] = 1;
		esa_y = y;
		esa_x = x;
	};

	snakegame(int size, int hp) {
		//コンストラクタ
		HP = hp;
		score = 0;
		tablesize = size;
		Heal = size * 4;
		snake_y = Random(tablesize - 1);
		snake_x = Random(tablesize - 1);
		position.emplace_front(make_pair(snake_x, snake_y));

		table.resize(tablesize);
		for (int i = 0; i < tablesize; i++) {
			table[i].resize(tablesize);
			for (int j = 0; j < tablesize; j++)table[i][j] = 0;
		}
		generate();
	};

	void init() {
		//盤面初期化
		for (int i = 0; i < tablesize; i++) {
			for (int j = 0; j < tablesize; j++) {
				table[i][j] = 0;
			}
		}
		score = 0;
		HP = tablesize * 2;
		snake_x = Random(tablesize - 1);
		snake_y = Random(tablesize - 1);
		position.clear();
		position.emplace_front(make_pair(snake_x, snake_y));
		generate();

	};
	void move(int dirction) {
		//0:上,1:右,2:下,3:左
		score++;
		dirction = dirction % 4;//一応あまりを取っておいて範囲内に収まるようにする
		int diry[] = { -1,0,1,0 };
		int dirx[] = { 0,1,0,-1 };
		int y = snake_y + diry[dirction];
		int x = snake_x + dirx[dirction];
		HP--;
		if (x < 0 || TABLESIZE <= x || y < 0 || TABLESIZE <= y) {
			HP -= 15;
			return;
		}
		snake_y = y;
		snake_x = x;
		position.emplace_front(make_pair(snake_x, snake_y));
		while (position.size() > 6)position.pop_back();
	}
	bool gameover() {
		if (HP <= 0)return true;
		else return false;
	}
};

class Neuron {
public:
	int n;
	double count;//worningが出る
	double u, eta, alpha, delta;
	vector <double> w;
	double Random() {
		int now = rand();
		double ret = now;
		while (ret > 1)ret = ret / 10.0;
		return ret;
	}
	void init(int num)
	{
		int i;
		n = num;
		alpha = 2.0;
		eta = 0.05;
		w.resize(n);
		//乱数(-1.000 ~ 1.000)で初期化
		for (i = 0; i < n; i++)w[i] = Random();
	};
	void reset() {
		for (int i = 0; i < n; i++)w[i] = Random();
	};
	double calc(vector<double> x)
	{
		int i;
		double ret;
		u = 0;
		for (i = 0; i < n; i++)u += x[i] * w[i];
		ret = 1.0 / (1.0 + exp(-alpha * u));
		return ret;
	};
	void learn(vector<double> x, double out, double sumdelta)
	{
		int i;
		delta = alpha * out * (1 - out) * sumdelta;
		for (i = 0; i < n; i++)
		{
			w[i] = w[i] + eta * delta * x[i];
		}
		return;
	};
};

class NNetwork {
public:
	vector<vector<Neuron>> net;//ネットワーク
	vector<vector<double>> table;//計算結果を入れておくところ
	void init(int entry = 2, int end = 1, int depth = 1, int width = 2) {
		//入力の数、出力の幅、中間層の深さ、層の幅
		srand(time(NULL));
		for (int i = 0; i < depth; i++)net.push_back(vector<Neuron>(width));

		net.push_back(vector<Neuron>(end));//出力層を追加
		table.resize(net.size());
		for (int i = 0; i < net.size(); i++) {
			for (int j = 0; j < net[i].size(); j++) {
				table[i].push_back(0.0);
			}
		}

		for (int i = 0; i < net.size(); i++) {
			for (int j = 0; j < net[i].size(); j++) {
				if (i == 0)net[i][j].init(entry);//入力層だけ別で処理
				else net[i][j].init(width);
			}
		}
	};

	vector<double> calc(vector<double> input) {
		//出力層の出力をそのまま返す
		for (int i = 0; i < table.size(); i++) {
			for (int j = 0; j < table[i].size(); j++) {
				if (i == 0)table[i][j] = net[i][j].calc(input);
				else table[i][j] = net[i][j].calc(table[i - 1]);
			}
		}
		vector<double> ret;
		for (int i = 0; i < table[table.size() - 1].size(); i++)ret.push_back(table[table.size() - 1][i]);
		return ret;//最後の出力が出力層の出力
	};
	void learn(vector<double> ans, vector<double> inp) {
		calc(inp);
		if (ans.size() != table[table.size() - 1].size())return;//出力層のサイズと教師信号が違ったら駄目
		for (int i = net.size() - 1; i >= 0; i--) {
			for (int j = 0; j < net[i].size(); j++) {
				if (i == 0) {
					for (int k = 0; k < net[i + 1].size(); k++) {
						net[i][j].learn(inp, table[i][j], net[i + 1][k].delta * net[i + 1][k].w[j]);
					}
				}
				else if (i == table.size() - 1)net[i][j].learn(table[i - 1], table[i][j], ans[j] - table[i][j]);
				else for (int k = 0; k < net[i + 1].size(); k++)net[i][j].learn(table[i - 1], table[i][j], net[i + 1][k].delta * net[i + 1][k].w[j]);
			}
		}
	};
};

class GA {
public:
	int popusize;//個体数
	double possibility;//突然変異する可能性
	int now;//現在の個体
	int dep;//深さ
	int wid;//幅
	int entrysiz;//入力サイズ
	int endsiz;//出力サイズ
	vector<NNetwork> individual;//個体
	vector<int> value;//評価値(生き残ったターン数)

	//コンストラクタ
	GA(int siz = 200, double possi = 0.1, int entry = 1, int end = 1, int depth = 3, int width = 10) {
		popusize = siz;
		possibility = possi;
		entrysiz = entry;
		endsiz = end;
		dep = depth;
		wid = width;
		value.resize(popusize);
		individual.resize(popusize);
		for (int i = 0; i < popusize; i++)individual[i].init(entrysiz, endsiz, dep, wid);
		fill(value.begin(), value.end(), 0.0);
		now = 0;
	};
	vector<double> get(vector<vector<int>> table, int snakey, int snakex) {
		//盤面と現在地からネットワークへの入力を求める
		vector<double> reply;
		int h = table.size();
		if (h == 0)return reply;//例外処理
		int w = table[0].size();
		//八方向のベクトル

		int dirx[] = { 0,  1,  0,-1 };
		int diry[] = { -1,  0,  1,  0 };
		for (int d = 0; d < 4; d++) {
			//餌を探索
			bool flg = true;
			int y = snakey;
			int x = snakex;
			while (1) {
				y += diry[d];
				x += dirx[d];
				if (y < 0 || h <= y || x < 0 || w <= x)break;
				double dist = abs(snakey - y) + abs(snakex - x);
				if (table[y][x] == 1) {
					if (d == 0 || d == 2) {
						//上下
						flg = false;
						reply.push_back(1.0 - (dist - 1) / ((double)(h - 1)));
						break;
					}
					else {
						//左右
						flg = false;
						reply.push_back(1.0 - (dist - 1) / ((double)(w - 1)));
					}
				}
			}
			if (flg)reply.push_back(0.0);
		}
		for (int d = 0; d < 4; d++) {
			//壁を探索
			bool flg = true;
			int y = snakey;
			int x = snakex;
			while (1) {
				y += diry[d];
				x += dirx[d];
				if (y < 0 || h <= y || x < 0 || w <= x) {
					//壁に衝突
					double dist = abs(snakey - y) + abs(snakex - x);
					if (d == 0 || d == 2)reply.push_back(1.0 - (dist - 1) / ((double)(h - 1)));//上下
					else reply.push_back(1.0 - (dist - 1) / ((double)(w - 1)));//左右
					break;
				}
			}
		}
		return reply;
	}
	bool ended() {
		//すべての個体の評価が終了したかどうか
		//false:終わってない
		//true:すべての個体が終わった
		if (now < popusize)return false;
		else return true;
	}
	void evaluate(int val) {
		value[now] = val;
		now++;
	}
	int getoutput(vector<vector<int>> table, int snakey, int snakex) {
		vector<double> inp = get(table, snakey, snakex);
		vector<double> out = individual[now].calc(inp);
		double ma = 0;
		int ind = 0;
		for (int i = 0; i < out.size(); i++) {
			if (ma < out[i]) {
				ind = i;
				ma = out[i];
			}
		}
		return ind;
	}
	void evol() {
		now = 0;
		fill(value.begin(), value.end(), 0.0);//評価値を初期化
	}
	void Inherit() {
		//継承させる
		now = 0;
		int endindx = 0;
		int firstind = 0;
		int secondind = 0;
		//優秀な遺伝子を探す
		vector<pair<int, int>> ind;
		for (int i = 0; i < popusize; i++)ind.emplace_back(make_pair(value[i], i));
		sort(ind.begin(), ind.end());
		reverse(ind.begin(), ind.end());//評価値大きい順にならんでてpopusize/2からpopusizeまで選ぶ
		vector<NNetwork> now = individual;
		int plus = popusize / 2;
		for (int i = 0; i < plus; i++)individual[i] = now[ind[i].second];
		for (int i = 0; i < plus / 2; i++) {
			auto Extract = [](NNetwork sample) {
				//ネットワークから荷重を抽出する関数(ラムダ)
				vector<double> table;
				for (int j = 0; j < sample.net.size(); j++) {
					for (int k = 0; k < sample.net[j].size(); k++) {
						for (int l = 0; l < sample.net[j][k].w.size(); l++)table.push_back(sample.net[j][k].w[l]);
					}
				}
				return table;
			};
			vector<double> extractedA = Extract(now[ind[2 * i].second]);
			vector<double> extractedB = Extract(now[ind[2 * i + 1].second]);
			vector<double> A;
			vector<double> B;
			for (int j = 0; j < extractedA.size(); j++) {
				//荷重を混ぜる
				if (RandomBool(possibility)) {
					A.push_back(Random());
					B.push_back(Random());
				}
				else if (RandomBool()) {
					A.push_back(extractedA[j]);
					B.push_back(extractedB[j]);
				}
				else {
					A.push_back(extractedB[j]);
					B.push_back(extractedA[j]);
				}
			}
			int count = 0;
			//混ぜた荷重を合流させる
			for (int j = 0; j < individual[plus + 2 * i].net.size(); j++) {
				for (int k = 0; k < individual[plus + 2 * i].net[j].size(); k++) {
					for (int l = 0; l < individual[plus + 2 * i].net[j][k].w.size(); l++) {
						individual[plus + 2 * i].net[j][k].w[l] = A[count];
						individual[plus + 2 * i + 1].net[j][k].w[l] = B[count];
						count++;
						endindx = max(endindx, plus + 2 * i + 1);
					}
				}
			}
		}
		for (int i = endindx + 1; i < popusize; i++)individual[i].init(entrysiz, endsiz, dep, wid);
		fill(value.begin(), value.end(), 0.0);//評価値を初期化
	}
	void learn(vector<double> ans, vector<vector<int>> table, int snakey, int snakex) {
		vector<double> inp = get(table, snakey, snakex);
		individual[now].learn(ans, inp);
	}
};

void Main() {
	Window::Resize(Size(1200, 800));
	int handflg = 1;//0以外だったらAI

	const Font font(30, Typeface::Light);

	double hp = 50;//体力
	double sim = 20;//速度
	double tablesiz = 10;//盤面の大きさ


	int sedai = 1;
	int maxscore = 0;
	int beginy = 70;
	int beginx = 30;
	int beginry = 70;
	int beginrx = 860;

	//GA(siz,possi,entry,end,depth,width)
	GA test(1, 0.2, 8, 4, 4, 4);
	int refresh = 20;
	while (System::Update()) {
		//ステータス設定
		//break;
		SimpleGUI::Slider(U"snakes HP:{}"_fmt((int)hp), hp, 10, 100, Vec2(100, 100), 200, 180);
		SimpleGUI::Slider(U"tablesize:{}"_fmt((int)tablesiz), tablesiz, 10, 100, Vec2(100, 140), 200, 180);
		SimpleGUI::Slider(U"sim speed:{}"_fmt((int)sim), sim, 10, 360, Vec2(600, 600), 200, 350);
		if (SimpleGUI::Button(U"sim speed update", Vec2(600, 640))) {
			refresh = sim;
		}
		if (SimpleGUI::Button(U"start", Vec2(100, 220)))break;
		if (SimpleGUI::Button(U"play", Vec2(400, 220))) {
			handflg = 0;
			break;
		}
	}
	TABLESIZE = ((int)tablesiz);
	snakegame game(TABLESIZE, int(hp));
	Graphics::SetTargetFrameRateHz(refresh);//60Hz固定
	deque<int> scores;
	while (System::Update()) {
		//1フレームにかけられる秒数:0.0166666666666667[s](60Hz)
		//余裕を持って0.016sで処理を終わらせること！
		//盤面の背景
		Rect(beginx - 10, beginy - 10, (TABLESIZE - 1) * 20 - (TABLESIZE - 1) + 40, (TABLESIZE - 1) * 20 - (TABLESIZE - 1) + 40).draw(Palette::Teal);
		//ゲームオーバー処理
		maxscore = max(maxscore, game.score);
		if (game.gameover()) {
			test.evaluate(game.score);
			scores.push_back(game.score);
			for (int i = 50; i < scores.size(); i++)scores.pop_front();
			game.init();
			if (test.ended() == true) {
				sedai++;

				//test.Inherit();
				test.evol();
			}
		}
		int ou = test.getoutput(game.table, game.snake_y, game.snake_x);
		vector<double> are = test.get(game.table, game.snake_y, game.snake_x);
		if (handflg) {
			for (int i = 0; i < 4; i++) {
				break;
				//餌
				if (are[i] > 0 && ou != i) {
					vector<double> ans = { 0,0,0,0 };
					ans[i] = 1;
					test.learn(ans, game.table, game.snake_y, game.snake_x);
				}
			}
			if (are[4] == 1.0 && are[7] == 1.0) {
				//左上に壁がある
				//下にいかせる
				vector<double> ans = { 0,0,1,0 };
				test.learn(ans, game.table, game.snake_y, game.snake_x);
			}
			else if (are[4] == 1.0) {
				//上に壁がある
				//左にいかせる
				vector<double> ans = { 0,0,0,1 };
				test.learn(ans, game.table, game.snake_y, game.snake_x);
			}
			else if (are[5] == 1.0) {
				//右に壁がある
				//上にいかせる
				vector<double> ans = { 1,0,0,0 };
				test.learn(ans, game.table, game.snake_y, game.snake_x);
			}
			else if (are[6] == 1) {
				//下に壁がある
				//右にいかせる
				vector<double> ans = { 0,1,0,0 };
				test.learn(ans, game.table, game.snake_y, game.snake_x);
			}
			else if (are[7] == 1) {
				//左に壁がある
				//下にいかせる
				vector<double> ans = { 0,0,1,0 };
				test.learn(ans, game.table, game.snake_y, game.snake_x);
			}
			game.move(ou);
		}
		else {
			if (KeyUp.down())game.move(0);
			if (KeyRight.down())game.move(1);
			if (KeyDown.down())game.move(2);
			if (KeyLeft.down())game.move(3);
		}

		font(U"HP: {:0>2},maxscore:{},generation:{}"_fmt(game.HP, maxscore, sedai)).drawAt(Point(beginrx, beginry), Palette::White);//体力を描画
		font(U"nowout:{}"_fmt(ou)).drawAt(Point(beginrx, beginry + 50), Palette::White);

		SimpleGUI::Slider(U"sim speed:{}"_fmt((int)sim), sim, 10, 360, Vec2(600, 600), 200, 350);
		if (SimpleGUI::Button(U"sim speed update", Vec2(600, 640))) {
			refresh = sim;
			Graphics::SetTargetFrameRateHz(refresh);
		}

		double sam = 0;
		double param = 0;
		String s;
		for (int i = 0; i < scores.size(); i++) {
			sam += scores[i];
			param += 1;
		}
		if (param == 0)font(U"Average of the last 50 times:{}"_fmt(game.score)).drawAt(Point(beginrx, beginry + 100), Palette::White);
		else 	font(U"Average of the last 50 times:{:0>2.2f}"_fmt(sam / param)).drawAt(Point(beginrx, beginry + 100), Palette::White);
		//*/
		//マスと餌を描画,餌の当たり判定
		for (int i = 0; i < TABLESIZE; i++) {
			for (int j = 0; j < TABLESIZE; j++) {
				Rect(beginx + j * 20 - j, beginy + i * 20 - i, 20, 20).drawFrame(1, 0, Palette::Slategray);
				if (game.table[i][j] == 1) {
					if (game.snake_y == i && game.snake_x == j) {
						//蛇の居場所と餌が同じ座標なら
						game.table[i][j] = 0;//食べる
						game.HP += game.Heal;//食べたら回復
						game.generate();//餌を生成
					}
				}
			}
		}
		//諸々描画


		for (int i = 0; i < game.position.size(); i++) {
			int x = game.position[i].first;
			int y = game.position[i].second;
			if (x < 0 || TABLESIZE <= x || y < 0 || TABLESIZE <= y)continue;
			if (i != 0)Rect(beginx + x * 20 - x + 3, beginy + y * 20 - y + 3, 14, 14).draw(Color(0, 0, 0, 255 - i * (int)(255.0 / 10.0)));//過去にいた場所
		}
		Rect(beginx + game.snake_x * 20 - game.snake_x + 3, beginy + game.snake_y * 20 - game.snake_y + 3, 14, 14).draw(Palette::Red);//現在地;
		Rect(beginx + game.esa_x * 20 - game.esa_x + 3, beginy + game.esa_y * 20 - game.esa_y + 3, 14, 14).draw(Palette::Orange);//餌
	}
}
