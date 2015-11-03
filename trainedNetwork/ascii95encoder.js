
'use strict';

module.exports = ASCII95Encoder = {
	networkName: "ASCII95Encoder",
	networkType: "autoencoder",
	
	networkJSON: {"layers":[{"out_depth":8,"out_sx":1,"out_sy":1,"layer_type":"input"},{"out_depth":8,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":8,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":8,"w":{"0":-405.61349547678924,"1":2824.458870661135,"2":-2565.40446621382,"3":325.81519324863353,"4":1010.1227610566382,"5":-1463.3293284678973,"6":1004.1037338919165,"7":405.54155953485474}},{"sx":1,"sy":1,"depth":8,"w":{"0":17.6284631801829,"1":-159.74287821462371,"2":-130.1429488208937,"3":-86.73565836874192,"4":156.47698721530133,"5":-78.63113095689992,"6":157.68157294432737,"7":-17.924093592876723}},{"sx":1,"sy":1,"depth":8,"w":{"0":663.8779108300869,"1":-788.7530844532442,"2":130.33164460903512,"3":-133.03363840469797,"4":919.3004062274252,"5":-128.03900485530258,"6":1996.458000558927,"7":-664.0221624456137}},{"sx":1,"sy":1,"depth":8,"w":{"0":-516.2710165408953,"1":232.38105052882656,"2":-712.1045024614057,"3":228.50923902485422,"4":34.63000363299078,"5":2307.255365767692,"6":45.345498562705146,"7":516.2735853453404}},{"sx":1,"sy":1,"depth":8,"w":{"0":-45.961742158643204,"1":-282.3360788883221,"2":-277.7109210826698,"3":624.0274122120365,"4":205.70188931967016,"5":316.3916881990616,"6":369.48366536255486,"7":46.78669190239508}},{"sx":1,"sy":1,"depth":8,"w":{"0":272.40664245000147,"1":-570.8332256861685,"2":-203.30354387302756,"3":1058.7616864222025,"4":-2387.0504319773235,"5":845.2522011209533,"6":777.069386563223,"7":-272.68305348582816}},{"sx":1,"sy":1,"depth":8,"w":{"0":-9.398761340907572,"1":55.80544559601565,"2":103.0501763397419,"3":1361.2378033498721,"4":190.98452058503346,"5":-1160.6177482219734,"6":29.399977008074273,"7":9.552646197707471}},{"sx":1,"sy":1,"depth":8,"w":{"0":-641.0003798070811,"1":-1544.5472661407984,"2":-1707.3927713358184,"3":301.09001673555423,"4":230.81852713240215,"5":-939.9836289413449,"6":-1284.054888891539,"7":641.3873471881516}}],"biases":{"sx":1,"sy":1,"depth":8,"w":{"0":-406.7555168262446,"1":18.922836943190006,"2":668.6649243263617,"3":-518.7213336817659,"4":-46.172544662903206,"5":275.3222128917225,"6":-8.2281895445409,"7":-643.7229796045548}}},{"min_val":0,"max_val":1,"threshold":0.3,"out_depth":8,"out_sx":1,"out_sy":1,"layer_type":"step"},{"out_depth":127,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":8,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":8,"w":{"0":-13.2786558267446,"1":-16.317845513011413,"2":-16.759874579784064,"3":-5.649532549281051,"4":-16.314757466032787,"5":-14.003988271379753,"6":-16.343874394334605,"7":-12.393428269160967}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.329668383860523,"1":-16.37764050469605,"2":-16.851745657712378,"3":-5.687374711746291,"4":-16.331368183592634,"5":-14.017447248021143,"6":-16.412622206034452,"7":-12.43866726818247}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.299547629036054,"1":-16.33414544076837,"2":-16.81095528317772,"3":-5.665567506398028,"4":-16.311853456493146,"5":-14.008908624762574,"6":-16.38543727227605,"7":-12.412153627267965}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.301204022752108,"1":-16.347206706149045,"2":-16.78656925855785,"3":-5.666837846464217,"4":-16.315410764560276,"5":-14.009277777614977,"6":-16.379155574282272,"7":-12.41362712201965}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.321954505398022,"1":-16.35867458843182,"2":-16.840876391989926,"3":-5.6820678909272315,"4":-16.32564114773281,"5":-14.014963374838508,"6":-16.400359769159287,"7":-12.432036533023115}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.314358336708217,"1":-16.360274979892843,"2":-16.80670258896889,"3":-5.6778489284088955,"4":-16.298031821887502,"5":-14.011299264229801,"6":-16.418339869604356,"7":-12.42573561827422}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.28281445648386,"1":-16.318100801859422,"2":-16.766931354313396,"3":-5.653284816890076,"4":-16.302542769844965,"5":-14.004303077970504,"6":-16.359440341536878,"7":-12.397390232457755}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.309361696240158,"1":-16.339735422957332,"2":-16.812842916414432,"3":-5.673509142764766,"4":-16.31950004403523,"5":-14.01080618099434,"6":-16.38439292031353,"7":-12.421139621199188}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.314560172277856,"1":-16.355506755023693,"2":-16.82756314876748,"3":-5.676612417177829,"4":-16.321613848344284,"5":-14.012978234835582,"6":-16.398671813955975,"7":-12.425429677013923}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.308715889303753,"1":-16.357809124195206,"2":-16.812177221294135,"3":-5.671590926150639,"4":-16.311901068496624,"5":-14.012089894788522,"6":-16.39768249516879,"7":-12.420025261839445}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.323879373739043,"1":-16.383398772012878,"2":-16.84918302977342,"3":-5.682263856377321,"4":-16.326677057276587,"5":-14.016772397786892,"6":-16.419428374385347,"7":-12.433163291445916}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.290310483717442,"1":-16.339745205837033,"2":-16.781251359953867,"3":-5.658538306915352,"4":-16.305968777731475,"5":-14.006627732573687,"6":-16.37951013350572,"7":-12.403828124536701}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.341816274132103,"1":-16.378893059687602,"2":-16.873083727716395,"3":-5.6979449892113525,"4":-16.328040238748535,"5":-14.018980399360561,"6":-16.43461506456759,"7":-12.449991502305435}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.305969490336258,"1":-16.339599237758385,"2":-16.802036486692856,"3":-5.671463867953496,"4":-16.30579622409387,"5":-14.009333583668201,"6":-16.393941564729804,"7":-12.418279188061941}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.300441147730383,"1":-16.317921793448562,"2":-16.787744945824983,"3":-5.669040318776695,"4":-16.3003498484945,"5":-14.006076903310967,"6":-16.3877556331735,"7":-12.413972676579693}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.31465870531351,"1":-16.355696796499945,"2":-16.828941395561497,"3":-5.676645449546053,"4":-16.32073405114497,"5":-14.01304071894241,"6":-16.400871054139077,"7":-12.425499360495154}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.30598077621042,"1":-16.32427058134721,"2":-16.7993221750235,"3":-5.672867136732059,"4":-16.2997035731565,"5":-14.007821587447921,"6":-16.396234517879506,"7":-12.418815147354813}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.30490881997347,"1":-16.33811013608724,"2":-16.805025561105452,"3":-5.670397170594791,"4":-16.31557409918512,"5":-14.009419884194596,"6":-16.38589982602607,"7":-12.417208693812169}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.293169322667052,"1":-16.333752643285194,"2":-16.792434870749304,"3":-5.659997657831222,"4":-16.323514954784542,"5":-14.008179068050047,"6":-16.359712431062092,"7":-12.406204597297417}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.332231541530498,"1":-16.372334111989385,"2":-16.857918734629518,"3":-5.689635930747608,"4":-16.32050651800254,"5":-14.017664330505701,"6":-16.420827271517907,"7":-12.441154448393684}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.32322380295617,"1":-16.353725569303702,"2":-16.831655079774094,"3":-5.68537295814351,"4":-16.312085713978753,"5":-14.012732668452534,"6":-16.42174461337063,"7":-12.433911072304537}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.318360528083827,"1":-16.363520288026546,"2":-16.82590862098362,"3":-5.679656400712305,"4":-16.320381248153502,"5":-14.013714281457666,"6":-16.407482888004367,"7":-12.428854231711808}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.301792908494372,"1":-16.35773639599365,"2":-16.808255547797376,"3":-5.666095757623804,"4":-16.319007739750734,"5":-14.010728945731685,"6":-16.39106443723139,"7":-12.413645170447448}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.299628281239901,"1":-16.362819510710984,"2":-16.791470626725353,"3":-5.66550640694039,"4":-16.301208331735438,"5":-14.008968716588958,"6":-16.410815773302833,"7":-12.41201279613095}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.313101268946692,"1":-16.340833771390212,"2":-16.811242820023068,"3":-5.676905275555192,"4":-16.314287920161597,"5":-14.011085402035696,"6":-16.391103786428285,"7":-12.42470540069712}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.329231132361251,"1":-16.365251985685248,"2":-16.854205256733298,"3":-5.687704407659459,"4":-16.33521978581626,"5":-14.016681782558832,"6":-16.40542684178525,"7":-12.438536256950705}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.291125511173056,"1":-16.337763723508104,"2":-16.779088229517427,"3":-5.658712430941622,"4":-16.30409334772461,"5":-14.007261523460888,"6":-16.374772686005457,"7":-12.404485827358978}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.317762408344471,"1":-16.35814753833718,"2":-16.82230768239032,"3":-5.6789408235208745,"4":-16.310357774329844,"5":-14.013762085834134,"6":-16.40272447045401,"7":-12.428368559481096}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.356993286516843,"1":-16.380417946869364,"2":-16.907508402421243,"3":-5.709655230707836,"4":-16.322529282365597,"5":-14.022428275714343,"6":-16.455045523382612,"7":-12.463735119554034}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.32121999919306,"1":-16.366370377563633,"2":-16.8314868817841,"3":-5.682394229123624,"4":-16.333973851467235,"5":-14.013926047803217,"6":-16.404186248345542,"7":-12.431514852208351}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.310342131587521,"1":-16.36331802292474,"2":-16.82143055541135,"3":-5.672955553983828,"4":-16.32039757741435,"5":-14.012426350334758,"6":-16.404103819394294,"7":-12.4214053056004}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.304859037374985,"1":-16.341005819184435,"2":-16.799049691490673,"3":-5.670793105765458,"4":-16.302843690293105,"5":-14.008863387408007,"6":-16.398780547382063,"7":-12.417313836376823}},{"sx":1,"sy":1,"depth":8,"w":{"0":-23.037305261765155,"1":-11.98845219083853,"2":-40.59020145939069,"3":-14.171536292837622,"4":-19.837960973014848,"5":-11.548694042690391,"6":-25.225163388312662,"7":-8.442351260892625}},{"sx":1,"sy":1,"depth":8,"w":{"0":-21.294340747786606,"1":-14.890252279736724,"2":-49.951073378071094,"3":-3.482533095527004,"4":-20.660372392098534,"5":-12.281455436454672,"6":-27.73792487505612,"7":-6.528308399611553}},{"sx":1,"sy":1,"depth":8,"w":{"0":-24.027075171162057,"1":-18.621507004016102,"2":-19.764825179607385,"3":-15.52265790416691,"4":-17.07156672391143,"5":-24.619231162841465,"6":-22.036364130373208,"7":-14.03457886211801}},{"sx":1,"sy":1,"depth":8,"w":{"0":-21.819965635091993,"1":-20.182092689325362,"2":-20.652559712899243,"3":-5.055889592236717,"4":-18.68259112130051,"5":-25.10481174160006,"6":-18.937617189082978,"7":-14.527862312814878}},{"sx":1,"sy":1,"depth":8,"w":{"0":-30.413437888078686,"1":-15.33593682610342,"2":-44.576556801656515,"3":-13.90394000908936,"4":-13.126784426204651,"5":-12.81546865104457,"6":-15.25565215955616,"7":-10.388835303351331}},{"sx":1,"sy":1,"depth":8,"w":{"0":-23.578395644042004,"1":-29.827148567707716,"2":-40.576895304810456,"3":-5.396505974147084,"4":-5.742206010996914,"5":-12.594778023220026,"6":-28.65531216251792,"7":-6.449659967402545}},{"sx":1,"sy":1,"depth":8,"w":{"0":-17.534230292287997,"1":-19.37257831825401,"2":-22.189220130513323,"3":-11.442355840682553,"4":-13.87253519444257,"5":-24.165946775168116,"6":-16.35441368833527,"7":-15.924544995932546}},{"sx":1,"sy":1,"depth":8,"w":{"0":-28.918578365266803,"1":-18.233019150090722,"2":-23.708681061487542,"3":-9.391469487354834,"4":-13.685201577542001,"5":-16.342285677286614,"6":-16.06459698917681,"7":-13.458840560181531}},{"sx":1,"sy":1,"depth":8,"w":{"0":-20.309086713691545,"1":-26.13505628814591,"2":-19.90460157767846,"3":-12.878238282867954,"4":-14.307211841891876,"5":-18.473028506730305,"6":-20.959891974891022,"7":-16.06945248120758}},{"sx":1,"sy":1,"depth":8,"w":{"0":-18.680271894713968,"1":-24.12081154525593,"2":-26.27172369566651,"3":-13.136098668414744,"4":-14.455237375791934,"5":-15.361712454454572,"6":-20.496086586383814,"7":-15.923401841376483}},{"sx":1,"sy":1,"depth":8,"w":{"0":-25.619903363423862,"1":-18.434727788600057,"2":-19.717025082891716,"3":-16.11387463226961,"4":-23.336946090572105,"5":-26.036260203618017,"6":-16.50112922337258,"7":-14.951966293814544}},{"sx":1,"sy":1,"depth":8,"w":{"0":-23.732459686681224,"1":-16.970511063994365,"2":-16.214410506128327,"3":-16.881754554066027,"4":-19.017115293693088,"5":-31.542439091246532,"6":-24.416789262506796,"7":-30.131695975893453}},{"sx":1,"sy":1,"depth":8,"w":{"0":-24.249432063086932,"1":-28.461395126279008,"2":-38.555107788382,"3":-16.082683797994804,"4":-22.495854715933824,"5":-12.546655796065628,"6":-10.668799343798101,"7":-6.649196304983499}},{"sx":1,"sy":1,"depth":8,"w":{"0":-22.15877176238051,"1":-28.384284669349107,"2":-40.06654740457212,"3":-19.3929398808789,"4":-6.9773173503804235,"5":-11.676202797988217,"6":-12.18519320678733,"7":-21.62854380699439}},{"sx":1,"sy":1,"depth":8,"w":{"0":-31.788706623238152,"1":-25.291402749049016,"2":-19.155965319824325,"3":-13.655757786173516,"4":-12.901049566440516,"5":-26.659303222405853,"6":-12.16906816294341,"7":-12.516599231743487}},{"sx":1,"sy":1,"depth":8,"w":{"0":-39.73657482966865,"1":-40.18589028376714,"2":-12.361101119923775,"3":-19.918688031243665,"4":-7.781347835193267,"5":-38.46402680771167,"6":-13.198623153608107,"7":-21.825240401392687}},{"sx":1,"sy":1,"depth":8,"w":{"0":-5.286466735981011,"1":-25.59592675490375,"2":-37.47683716784769,"3":-16.108111554381882,"4":-16.72278368607717,"5":-13.004726031542127,"6":-26.11595490167709,"7":-18.265471497557705}},{"sx":1,"sy":1,"depth":8,"w":{"0":-10.76762122443818,"1":-24.711653028319354,"2":-41.618122630872534,"3":-3.3089275797840982,"4":-20.262155193892244,"5":-12.397064818253248,"6":-25.421596262460497,"7":-12.349189666362574}},{"sx":1,"sy":1,"depth":8,"w":{"0":-12.104840394108647,"1":-19.79679690122421,"2":-24.128850185257715,"3":-15.14473310416124,"4":-23.47379789080186,"5":-24.97575638847829,"6":-27.38848303482413,"7":-15.84207958956899}},{"sx":1,"sy":1,"depth":8,"w":{"0":-11.78451156842941,"1":-25.507366046034352,"2":-18.853503802014842,"3":-9.256304026627093,"4":-18.41974142178138,"5":-32.14137090636845,"6":-34.87767734961704,"7":-25.652162489213854}},{"sx":1,"sy":1,"depth":8,"w":{"0":-7.571062930306035,"1":-32.00125411738511,"2":-44.25961830998161,"3":-13.986012391998256,"4":-28.174341201231307,"5":-13.78717383006405,"6":-12.981249792622295,"7":-10.383457940148147}},{"sx":1,"sy":1,"depth":8,"w":{"0":-8.19344511791529,"1":-28.630106480574994,"2":-42.121840824100154,"3":-8.487039423219606,"4":-8.081572732473297,"5":-13.564352925567455,"6":-23.12222979167083,"7":-24.517861749943048}},{"sx":1,"sy":1,"depth":8,"w":{"0":-14.493199714534368,"1":-24.785655920774026,"2":-32.659307085593284,"3":-9.91989324059928,"4":-18.47001657804494,"5":-22.77901836014952,"6":-14.304589007910493,"7":-7.574614113521343}},{"sx":1,"sy":1,"depth":8,"w":{"0":-9.705815053966663,"1":-30.311145179386358,"2":-44.65841175007626,"3":-7.874893376446908,"4":-12.888356737018815,"5":-27.434750192898246,"6":-14.523913152246443,"7":-26.169765602872044}},{"sx":1,"sy":1,"depth":8,"w":{"0":-19.682864129974017,"1":-26.328571204837622,"2":-35.37765215668666,"3":-9.98895700640538,"4":-16.55111204427394,"5":-26.913970089961616,"6":-24.20370340722868,"7":-18.150082605400442}},{"sx":1,"sy":1,"depth":8,"w":{"0":-21.170300582497702,"1":-20.210350807393034,"2":-44.65651267353677,"3":-11.184635832016876,"4":-15.35881034237773,"5":-9.283326923689763,"6":-23.802284770972814,"7":-15.251189517092932}},{"sx":1,"sy":1,"depth":8,"w":{"0":-21.529766721329835,"1":-30.750073423294157,"2":-16.59214216740285,"3":-9.513332235366098,"4":-17.17006328480403,"5":-31.40577760735294,"6":-17.460005479088977,"7":-16.499250634979137}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.32545829936366,"1":-16.38480088374696,"2":-16.83760056006484,"3":-5.684273026700852,"4":-16.316096513718104,"5":-14.016219671931331,"6":-16.426887425779366,"7":-12.434887745575697}},{"sx":1,"sy":1,"depth":8,"w":{"0":-22.555589047512885,"1":-19.750344359880955,"2":-40.89299937918601,"3":-20.241168832973223,"4":-22.03917890471468,"5":-11.491091657408448,"6":-7.679883758983822,"7":-21.478621589802696}},{"sx":1,"sy":1,"depth":8,"w":{"0":-20.180973872210536,"1":-21.57969802760691,"2":-39.80941172952315,"3":-7.564941705328707,"4":-21.975216430374036,"5":-12.566963921239221,"6":-11.184684656377417,"7":-18.468014483310423}},{"sx":1,"sy":1,"depth":8,"w":{"0":-15.714035682894726,"1":-25.903819868703866,"2":-19.730717686881103,"3":-8.40841569525088,"4":-18.45831466959378,"5":-26.021708495349827,"6":-17.604040424286204,"7":-17.22336285265183}},{"sx":1,"sy":1,"depth":8,"w":{"0":-34.82960364110853,"1":-34.2502160001857,"2":-17.573191151828528,"3":-9.709760439546004,"4":-12.469650516894084,"5":-35.15441954809117,"6":-13.91914870022084,"7":-18.010143438833292}},{"sx":1,"sy":1,"depth":8,"w":{"0":-23.441793351135807,"1":-10.037707307854434,"2":-42.62510316662982,"3":-15.71356852896898,"4":-20.895352680567065,"5":-12.547720760641898,"6":-27.23571835185752,"7":-9.326007469894152}},{"sx":1,"sy":1,"depth":8,"w":{"0":-20.41269725011218,"1":-16.53894092980769,"2":-23.614466176922494,"3":-12.315015010571535,"4":-19.60516220952796,"5":-10.450758678351821,"6":-21.777103197203985,"7":-9.60179590626445}},{"sx":1,"sy":1,"depth":8,"w":{"0":-21.143543310396243,"1":-15.051169811211249,"2":-49.08119434294061,"3":-4.458628889661018,"4":-19.738829436206806,"5":-12.300019855383087,"6":-27.58109036336573,"7":-6.846757239884617}},{"sx":1,"sy":1,"depth":8,"w":{"0":-22.136823395408996,"1":-13.406417971248183,"2":-21.269225041634858,"3":-5.97531008359444,"4":-16.33086489279182,"5":-14.497309652106292,"6":-24.798534847524735,"7":-22.249592547842052}},{"sx":1,"sy":1,"depth":8,"w":{"0":-23.613998719937694,"1":-18.623884023148196,"2":-20.0528491385039,"3":-15.086186306975131,"4":-17.04095185951614,"5":-24.55934953641874,"6":-21.798085887332927,"7":-14.33354180980731}},{"sx":1,"sy":1,"depth":8,"w":{"0":-12.532058139430377,"1":-19.372531237932964,"2":-20.185228861382324,"3":-15.32942275983985,"4":-16.405713672437294,"5":-24.533149278888647,"6":-27.048850806946117,"7":-16.943391444653535}},{"sx":1,"sy":1,"depth":8,"w":{"0":-21.816260998369387,"1":-20.286763854967947,"2":-20.3686952893208,"3":-5.0067031503866275,"4":-18.674486423545652,"5":-25.169523882427992,"6":-18.899295996183323,"7":-14.490536865706488}},{"sx":1,"sy":1,"depth":8,"w":{"0":-24.185225660613223,"1":-18.144924714311916,"2":-20.116167646723,"3":-6.007497152314414,"4":-7.846609276734194,"5":-28.4287962798408,"6":-22.535498364434872,"7":-21.764852976169425}},{"sx":1,"sy":1,"depth":8,"w":{"0":-30.53447915603486,"1":-15.999590439746134,"2":-44.44144776444795,"3":-14.020743019258082,"4":-12.60253567750673,"5":-12.829092961815602,"6":-15.369431079800314,"7":-10.206317684319739}},{"sx":1,"sy":1,"depth":8,"w":{"0":-16.72812027859185,"1":-18.589365961467344,"2":-21.821888366213237,"3":-15.71497719424557,"4":-14.238799642563687,"5":-13.081795108052464,"6":-17.7836787507788,"7":-14.630371715164702}},{"sx":1,"sy":1,"depth":8,"w":{"0":-23.584297170657184,"1":-29.84165442791465,"2":-40.558274868547464,"3":-5.33741642858173,"4":-5.902841121529028,"5":-12.725465610816678,"6":-28.703849678233222,"7":-6.367753159718497}},{"sx":1,"sy":1,"depth":8,"w":{"0":-21.21919590630939,"1":-17.89675102489123,"2":-21.78551689372228,"3":-10.052316163235881,"4":-8.720359350402303,"5":-15.856524231494836,"6":-26.90025489184069,"7":-19.54773588362318}},{"sx":1,"sy":1,"depth":8,"w":{"0":-17.366394666345535,"1":-19.55143827795011,"2":-21.328600169399074,"3":-11.360694982369068,"4":-13.90322632885458,"5":-24.167280988680247,"6":-16.427595000278003,"7":-15.929023103854104}},{"sx":1,"sy":1,"depth":8,"w":{"0":-15.63675832374468,"1":-16.38662476971285,"2":-23.49703834010844,"3":-16.399443590876245,"4":-15.757662058807691,"5":-12.729397742568517,"6":-17.77682539032327,"7":-16.0975131463464}},{"sx":1,"sy":1,"depth":8,"w":{"0":-28.857051479218547,"1":-18.362683642672184,"2":-24.018721365172293,"3":-9.193993797865536,"4":-13.515394851279555,"5":-16.60793353510311,"6":-16.19266060930508,"7":-13.495215540657119}},{"sx":1,"sy":1,"depth":8,"w":{"0":-25.492509276755747,"1":-16.621293725097228,"2":-22.275219978872442,"3":-8.606213639745345,"4":-13.35703522743075,"5":-17.79350070604191,"6":-12.445466594621383,"7":-24.1356889557826}},{"sx":1,"sy":1,"depth":8,"w":{"0":-20.186353826692038,"1":-26.0015458108053,"2":-20.51682416408659,"3":-12.839811461676118,"4":-14.313275215061034,"5":-18.43133131027903,"6":-20.890209467069905,"7":-16.025680006218685}},{"sx":1,"sy":1,"depth":8,"w":{"0":-20.12673048352316,"1":-11.024843104987043,"2":-21.918148684527974,"3":-16.053324821078345,"4":-18.62809563275493,"5":-15.4572283958162,"6":-21.578070271511922,"7":-25.240219362869975}},{"sx":1,"sy":1,"depth":8,"w":{"0":-18.56971752253299,"1":-23.98454372162737,"2":-26.650091337147682,"3":-13.131776122646002,"4":-14.385557792630523,"5":-15.210201270490085,"6":-20.526685166547885,"7":-15.835528425218278}},{"sx":1,"sy":1,"depth":8,"w":{"0":-20.16681987738136,"1":-11.175564285878461,"2":-22.06060980408159,"3":-16.087699599568605,"4":-18.558453845510342,"5":-15.916593924472771,"6":-21.768014427741736,"7":-25.443980310633296}},{"sx":1,"sy":1,"depth":8,"w":{"0":-26.76777352614677,"1":-18.413922442159602,"2":-18.949624815548787,"3":-14.914079833446317,"4":-22.201811411664604,"5":-26.323309955656697,"6":-16.045593982530402,"7":-16.387040593049583}},{"sx":1,"sy":1,"depth":8,"w":{"0":-27.032981669911663,"1":-16.463350154452794,"2":-18.448663897209684,"3":-16.01261519668725,"4":-19.398404560047627,"5":-29.161306008205255,"6":-15.41411999172715,"7":-28.610366186725233}},{"sx":1,"sy":1,"depth":8,"w":{"0":-23.724870245538937,"1":-16.49146440526944,"2":-16.486804364905563,"3":-16.96099508392336,"4":-19.051108833318803,"5":-31.186017587491847,"6":-24.264417994102285,"7":-30.22340568183562}},{"sx":1,"sy":1,"depth":8,"w":{"0":-22.176608171925362,"1":-17.21230796723134,"2":-18.18420679207965,"3":-17.802470045092196,"4":-9.372224959918377,"5":-28.575134093505507,"6":-24.811308033192546,"7":-27.93972751021922}},{"sx":1,"sy":1,"depth":8,"w":{"0":-24.38329704949833,"1":-28.19102306550819,"2":-38.40654258177339,"3":-16.458994278045725,"4":-22.782628210196922,"5":-12.581715076157944,"6":-10.123047701358434,"7":-7.013221899226259}},{"sx":1,"sy":1,"depth":8,"w":{"0":-25.905152104224903,"1":-16.760638478481876,"2":-22.698091825919057,"3":-19.63724136376727,"4":-10.122300345599601,"5":-14.759051785081958,"6":-11.30851961900985,"7":-23.636548723019448}},{"sx":1,"sy":1,"depth":8,"w":{"0":-22.195363281961782,"1":-28.33695663764471,"2":-40.09658630327768,"3":-19.49574071251994,"4":-7.147578131253435,"5":-11.831196312876631,"6":-12.129025150771742,"7":-21.71672015484883}},{"sx":1,"sy":1,"depth":8,"w":{"0":-23.094366713868382,"1":-32.45008881131556,"2":-19.647085467453394,"3":-12.092940621670774,"4":-2.154387027756862,"5":-19.127156773250647,"6":-16.811864396132023,"7":-18.281204163002552}},{"sx":1,"sy":1,"depth":8,"w":{"0":-31.78263456012054,"1":-25.356673570044755,"2":-18.92286719755298,"3":-13.664046202125597,"4":-13.07888422568079,"5":-26.683058888666878,"6":-12.008298127705531,"7":-12.4728877747357}},{"sx":1,"sy":1,"depth":8,"w":{"0":-27.55993770839941,"1":-16.74101898923536,"2":-21.326853822240746,"3":-15.371065900088801,"4":-7.99066526848662,"5":-27.09329682955166,"6":-12.175340310114773,"7":-27.16730793018344}},{"sx":1,"sy":1,"depth":8,"w":{"0":-39.7933083588993,"1":-40.47950419177452,"2":-11.906479522307391,"3":-19.985202535099805,"4":-7.832962173283189,"5":-38.79051957230829,"6":-13.469658862135311,"7":-21.345667790784276}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.32218418065753,"1":-16.36591912607356,"2":-16.842262239543274,"3":-5.682339592511953,"4":-16.317779222126358,"5":-14.014874592338438,"6":-16.415899944111306,"7":-12.432213927976977}},{"sx":1,"sy":1,"depth":8,"w":{"0":-5.238411128104491,"1":-25.656069239911346,"2":-37.40355952844354,"3":-16.093646489942525,"4":-16.678455273898404,"5":-13.105302887409092,"6":-26.203708047016597,"7":-18.33034337917121}},{"sx":1,"sy":1,"depth":8,"w":{"0":-7.162385718005648,"1":-13.926343508548381,"2":-23.39862239956124,"3":-19.148875997118693,"4":-18.36806703379552,"5":-14.850805072320831,"6":-25.88486792109813,"7":-25.768707962527536}},{"sx":1,"sy":1,"depth":8,"w":{"0":-10.608306604923543,"1":-24.74725142711869,"2":-41.64534457523325,"3":-3.4415061014558756,"4":-20.285297876293253,"5":-12.516485953424833,"6":-25.33684582320311,"7":-12.571593934595336}},{"sx":1,"sy":1,"depth":8,"w":{"0":-9.127388139483828,"1":-20.291587981287236,"2":-21.73632521545088,"3":-5.7449653466370565,"4":-21.239816919902114,"5":-15.61657597784884,"6":-24.22219958367172,"7":-18.326339104933318}},{"sx":1,"sy":1,"depth":8,"w":{"0":-11.917518109437278,"1":-20.568372005271463,"2":-23.44274620383749,"3":-15.15801337104837,"4":-23.37171540685245,"5":-25.562386735289664,"6":-28.150143785519354,"7":-15.880501120536308}},{"sx":1,"sy":1,"depth":8,"w":{"0":-6.288041066012275,"1":-17.258015796706942,"2":-19.559431729103082,"3":-21.823691989173547,"4":-17.156361868982262,"5":-26.648963381885732,"6":-32.31466094570477,"7":-30.95575918111025}},{"sx":1,"sy":1,"depth":8,"w":{"0":-11.760197861576719,"1":-25.252730132741174,"2":-18.931463570004776,"3":-8.916180448941125,"4":-18.128163434978614,"5":-31.65932805705717,"6":-34.34620736912103,"7":-25.723849199976133}},{"sx":1,"sy":1,"depth":8,"w":{"0":-12.883705188670738,"1":-17.12794955604145,"2":-21.254330930070814,"3":-8.337650226942635,"4":-11.221563093363515,"5":-24.937712779575864,"6":-25.04228149431948,"7":-26.424730179700027}},{"sx":1,"sy":1,"depth":8,"w":{"0":-7.589159400557218,"1":-31.898252952255475,"2":-44.820556265558814,"3":-14.019137644090046,"4":-28.031248775917824,"5":-13.737099832026093,"6":-12.93439027700496,"7":-10.57447308222189}},{"sx":1,"sy":1,"depth":8,"w":{"0":-12.302081645179633,"1":-21.80746701555128,"2":-24.356724511561783,"3":-19.66772589220278,"4":-6.0074217529650715,"5":-13.878770075336387,"6":-17.51780726582105,"7":-29.09353748781953}},{"sx":1,"sy":1,"depth":8,"w":{"0":-8.325832558360765,"1":-28.13181794049986,"2":-42.372893645330564,"3":-8.311019646080996,"4":-8.096159862621103,"5":-13.316339779771073,"6":-22.83109841240786,"7":-25.471276270054662}},{"sx":1,"sy":1,"depth":8,"w":{"0":-9.537570417056191,"1":-30.146023099388276,"2":-22.895495703183975,"3":-6.972712013537999,"4":-7.143260834807392,"5":-17.981638658423854,"6":-24.359603773450676,"7":-22.981224839779898}},{"sx":1,"sy":1,"depth":8,"w":{"0":-14.483285907373645,"1":-24.84090753136283,"2":-33.33181248617682,"3":-9.53389803398017,"4":-18.319912437487502,"5":-22.543026517493032,"6":-14.224183896601644,"7":-7.893005610075011}},{"sx":1,"sy":1,"depth":8,"w":{"0":-14.40591107524169,"1":-16.54218077356164,"2":-21.21112417359726,"3":-18.04155232713373,"4":-10.248834719831744,"5":-25.255469247524292,"6":-18.352584678957875,"7":-32.58431518204865}},{"sx":1,"sy":1,"depth":8,"w":{"0":-9.553577000500396,"1":-30.333497818489395,"2":-44.61555725744212,"3":-7.76409827650535,"4":-12.805818879211973,"5":-27.36241604680689,"6":-14.455455389610096,"7":-26.146934644054283}},{"sx":1,"sy":1,"depth":8,"w":{"0":-12.952345326300538,"1":-19.17563619705845,"2":-23.490925749101752,"3":-8.054302788179708,"4":-11.550980113748906,"5":-16.661700977818896,"6":-17.665094509108815,"7":-26.88410400497668}},{"sx":1,"sy":1,"depth":8,"w":{"0":-19.696957462637485,"1":-26.324142280452328,"2":-35.359557031904274,"3":-10.037125094462498,"4":-16.557762365946924,"5":-26.904865637754607,"6":-24.195171291151386,"7":-18.13150767278851}},{"sx":1,"sy":1,"depth":8,"w":{"0":-3.3467101586617573,"1":-31.103843798408693,"2":-20.312911221423174,"3":-13.806878528005292,"4":-16.680779583809752,"5":-19.83338399647274,"6":-33.92725184579191,"7":-20.226253372663546}},{"sx":1,"sy":1,"depth":8,"w":{"0":-21.196578099456662,"1":-20.252298150320378,"2":-44.64575995596812,"3":-11.199633323086159,"4":-15.253006363652852,"5":-9.268010025371023,"6":-23.89507907842887,"7":-15.219766519901483}},{"sx":1,"sy":1,"depth":8,"w":{"0":-20.059115108307306,"1":-26.252738617663652,"2":-19.376400528572777,"3":-6.008749577762882,"4":-14.48666246087399,"5":-19.317077089476136,"6":-20.81405509072187,"7":-16.081467631084273}},{"sx":1,"sy":1,"depth":8,"w":{"0":-21.101091617494745,"1":-30.52925407251389,"2":-16.62773494643342,"3":-9.321685122782243,"4":-17.178869994085815,"5":-31.19309792056489,"6":-17.649998231805803,"7":-16.508247573185944}},{"sx":1,"sy":1,"depth":8,"w":{"0":-13.368023256117341,"1":-10.809276787551194,"2":-23.11388196033559,"3":-15.881578595663196,"4":-22.215949252329935,"5":-22.1716680231591,"6":-16.91785456635815,"7":-24.53195243635697}},{"sx":1,"sy":1,"depth":8,"w":{"0":-26.280216498530492,"1":-42.74736284985619,"2":-7.300922572029368,"3":-8.95439797228943,"4":-13.434313631078543,"5":-45.17960873528442,"6":-34.42494233764135,"7":-18.649154850931527}},{"sx":1,"sy":1,"depth":8,"w":{"0":-25.75770007755655,"1":-27.713827965675122,"2":-11.648612287526479,"3":-7.984989754082623,"4":-19.139448126932492,"5":-39.696690417692146,"6":-29.235036605918537,"7":-27.715313969066933}},{"sx":1,"sy":1,"depth":8,"w":{"0":-22.579967563320224,"1":-19.724184767154302,"2":-40.83019065010355,"3":-20.2507944502722,"4":-22.038441735282124,"5":-11.443170087927743,"6":-7.60300063233135,"7":-21.459685578005537}},{"sx":1,"sy":1,"depth":8,"w":{"0":-12.56094819852176,"1":-24.1375227887865,"2":-24.884728281226554,"3":-11.097080447480609,"4":-22.234359278249954,"5":-9.883383964298065,"6":-12.17174017997862,"7":-16.03884604737824}},{"sx":1,"sy":1,"depth":8,"w":{"0":-20.160783422594342,"1":-21.47823713831109,"2":-39.72865708774973,"3":-7.557638101518172,"4":-21.887898746109275,"5":-12.285629648987749,"6":-11.070054633354001,"7":-18.348177183912423}},{"sx":1,"sy":1,"depth":8,"w":{"0":-24.248220362973612,"1":-31.311445918231303,"2":-19.57746089790302,"3":-8.82823683491159,"4":-8.264898706132706,"5":-18.81600165474002,"6":-13.755326366921961,"7":-17.143517805849438}},{"sx":1,"sy":1,"depth":8,"w":{"0":-15.252937730719775,"1":-23.821377650698092,"2":-17.89857881067105,"3":-8.698076126950555,"4":-19.923265656038182,"5":-23.88666184671069,"6":-15.903071380349711,"7":-17.911544687261504}},{"sx":1,"sy":1,"depth":8,"w":{"0":-15.252892305834178,"1":-18.567096038412153,"2":-20.469217359730226,"3":-17.38395228785252,"4":-8.71090410867118,"5":-26.229645613745344,"6":-18.592260313412044,"7":-32.24735549465058}},{"sx":1,"sy":1,"depth":8,"w":{"0":-34.3438884166614,"1":-34.16304459963666,"2":-17.96275918763239,"3":-9.575533805978516,"4":-12.515619766534504,"5":-34.976219630921314,"6":-13.88290036509876,"7":-17.832100639455156}}],"biases":{"sx":1,"sy":1,"depth":127,"w":{"0":-45.21687338766447,"1":-44.872464406988506,"2":-45.06708263316986,"3":-45.07776720897562,"4":-44.93668070407078,"5":-44.99579522102882,"6":-45.1978240009364,"7":-45.032652036041526,"8":-44.974494802516816,"9":-45.01035508679786,"10":-44.87900948537871,"11":-45.12245783105289,"12":-44.80639858583982,"13":-45.05521908638009,"14":-45.11527820208691,"15":-44.97139183478767,"16":-45.077730601963104,"17":-45.05401177662612,"18":-45.113030751838124,"19":-44.86906689987595,"20":-44.94160489093408,"21":-44.952741367604396,"22":-45.02820226936809,"23":-45.043187857565286,"24":-45.02413373512655,"25":-44.88688989384227,"26":-45.131318757598876,"27":-44.977641669941846,"28":-44.7238161789728,"29":-44.92819102934444,"30":-44.97729161324078,"31":-45.05736403981567,"32":-15.589592515467556,"33":-14.625518019982934,"34":-20.71293016969036,"35":-21.225956676472805,"36":-16.81098424929733,"37":-13.491557982661307,"38":-20.534145687883352,"39":-21.9734610867403,"40":-19.066202813086146,"41":-15.780043827381147,"42":-19.313031468322414,"43":-16.593613467675276,"44":-13.309365969149874,"45":-12.485401329306141,"46":-19.684831180215575,"47":-12.803401190636478,"48":-13.987408478298716,"49":-13.160092197362917,"50":-16.127823378349493,"51":-14.024129589850082,"52":-17.19738474996217,"53":-14.483311734929039,"54":-16.663942878492335,"55":-14.89266238794291,"56":-3.1520761157011545,"57":-10.110293197726601,"58":-17.33249331008411,"59":-44.88948853941773,"60":-12.368702685394552,"61":-12.80533940987311,"62":-19.971852718260574,"63":-13.760948354091796,"64":-14.396316993557395,"65":-27.657172489033403,"66":-13.90127616389637,"67":-26.731422499362732,"68":-20.23082146420625,"69":-21.025729461322875,"70":-21.424691048174512,"71":-23.197404165509518,"72":-16.727044601174853,"73":-26.100518414094487,"74":-13.328632002215561,"75":-23.821322173791216,"76":-21.331002742600727,"77":-25.001184645498384,"78":-21.508978474078287,"79":-23.912002164389513,"80":-18.465686331538794,"81":-23.348355084083355,"82":-15.548673469306882,"83":-22.75423827970546,"84":-19.38506613177549,"85":-18.1299536347,"86":-16.740327700691182,"87":-18.542889088290152,"88":-13.464941747778509,"89":-25.86880548635597,"90":-12.283262399882245,"91":-22.30417470595163,"92":-19.92137411324047,"93":-22.903055944585656,"94":-12.84260558905503,"95":-44.920020718113086,"96":-13.881999512310725,"97":-26.29533857749354,"98":-13.060858079440878,"99":-26.853360306662122,"100":-16.153836674675066,"101":-21.354321216291346,"102":-14.556618179713371,"103":-22.947600914887562,"104":-17.055380025575832,"105":-25.687821997336932,"106":-14.775115579731277,"107":-23.702937285306763,"108":-16.410833496752364,"109":-22.400772239002496,"110":-15.280861049883732,"111":-25.309740765744017,"112":-3.080843552807059,"113":-21.334756460017253,"114":-10.072134702138598,"115":-20.698532406357504,"116":-17.580865380415478,"117":-24.005230474408023,"118":-11.881899966395716,"119":-14.122808946007172,"120":-12.483509713542975,"121":-28.31107141593611,"122":-13.154046282648458,"123":-22.892243490677383,"124":-23.978399546402343,"125":-21.589121681067066,"126":-13.625435407141736}}},{"out_depth":127,"out_sx":1,"out_sy":1,"layer_type":"softmax","num_inputs":127}]}
};